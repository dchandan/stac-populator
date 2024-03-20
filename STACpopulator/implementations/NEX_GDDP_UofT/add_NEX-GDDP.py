import argparse
import json
import logging
import os
from typing import Any, MutableMapping, NoReturn, Optional, Tuple, Union
from urllib.parse import urlparse

import cftime
import netCDF4 as nc
import pystac
from marble_client import MarbleClient
from pydantic import BaseModel, ConfigDict, Field
from pystac import STACValidationError
from pystac.extensions.datacube import DatacubeExtension
from requests.sessions import Session

from STACpopulator.cli import add_request_options, apply_request_options
from STACpopulator.extensions.datacube import DataCubeHelper
from STACpopulator.extensions.thredds import THREDDSExtension, THREDDSHelper
from STACpopulator.input import ErrorLoader, GenericLoader, THREDDSLoader
from STACpopulator.models import GeoJSONPolygon
from STACpopulator.populator_base import STACpopulatorBase
from STACpopulator.stac_utils import ncattrs_to_bbox, ncattrs_to_geometry

LOGGER = logging.getLogger(__name__)


class NEXGDDPProperties(BaseModel, validate_assignment=True):
    """Data model for NEX GDDP files."""

    scenario: str = Field(alias="nexgddp:experiment_id")
    cmip6_source_id: str = Field(alias="nexgddp:source_id")
    cmip6_institution_id: str = Field(alias="nexgddp:institution_id")
    variant_label: str = Field(alias="nexgddp:variant_label")
    variable_id: str = Field(alias="nexgddp:variable_id")
    institution: str = Field(alias="nexgddp:institution")
    frequency: str = Field(alias="nexgddp:frequency")
    version: str = Field(alias="nexgddp:version")
    cmip6_license: str = Field(alias="nexgddp:license")
    Conventions: str = Field(alias="nexgddp:Conventions")
    calendar: str = Field(alias="nexgddp:calendar")
    cmip_version: str = Field(default="CMIP6", alias="nexgddp:cmip_version")
    start_datetime: str
    end_datetime: str

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class NEXGDDPCMIP6populator(STACpopulatorBase):
    item_geometry_model = GeoJSONPolygon

    def __init__(
        self,
        stac_host: str,
        catalog_url: str,
        data_loader: GenericLoader,
        update: Optional[bool] = False,
        session: Optional[Session] = None,
        config_file: Optional[Union[os.PathLike[str], str]] = None,
        log_debug: Optional[bool] = False,
        add_magpie_item_links: Optional[bool] = False,
    ) -> None:
        """Constructor

        :param stac_host: URL to the STAC API
        :type stac_host: str
        :param data_loader: loader to iterate over ingestion data.
        """
        super().__init__(
            stac_host, data_loader, update=update, session=session, config_file=config_file, log_debug=log_debug
        )
        self.add_magpie_item_links = add_magpie_item_links
        self._catalog_url = catalog_url
        self._marble_host_node = self.__get_marble_host_node_for_data()

    def __get_marble_host_node_for_data(self):
        name = None
        client = MarbleClient()
        hostname = urlparse(self._catalog_url).hostname
        for node_name, node in client.nodes.items():
            if hostname in node.url:
                name = node_name

        if name == None:
            raise RuntimeError("Could not infer name of the host that contains the data")
        return name

    def get_item_temporal_information(self, item_opendap_url: str) -> Tuple[str, str, str]:
        """Get the temporal extents (start and end date) by opening the file via its OpenDAP URL.

        :param item_name: the file's OpenDAP URL
        :type item_name: str
        :return: A tuple of start and end datetimes encoded as strings and the time calendar
        :rtype: Tuple[str, str, str]
        """
        ncf = nc.Dataset(item_opendap_url, "r")
        calendar = ncf["time"].calendar
        units = ncf["time"].units
        st = cftime.num2date(ncf["time"][0], units, calendar).isoformat() + "Z"
        ed = cftime.num2date(ncf["time"][-1], units, calendar).isoformat() + "Z"
        return st, ed, calendar

    def create_stac_item(
        self, item_name: str, item_data: MutableMapping[str, Any]
    ) -> Union[None, MutableMapping[str, Any]]:
        """Creates the STAC item.

        :param item_name: name of the STAC item. Interpretation of name is left to the input loader implementation
        :type item_name: str
        :param item_data: dictionary like representation of all information on the item
        :type item_data: MutableMapping[str, Any]
        :return: _description_
        :rtype: MutableMapping[str, Any]
        """

        # Create the pystac Item with NEX GDDP properties:
        # first get the variable id from the item_name
        varid = item_name.split("_")[0]
        # Now include the variable id in the item_data["attributes"] fiedd so the pydantic model can pick it up
        item_data["attributes"]["variable_id"] = varid
        # Get the item's start and end date from the item name
        st, ed, calendar = self.get_item_temporal_information(item_data["access_urls"]["OPENDAP"])
        item_data["attributes"]["start_datetime"] = st
        item_data["attributes"]["end_datetime"] = ed
        item_data["attributes"]["calendar"] = calendar
        properties = NEXGDDPProperties(**item_data["attributes"]).model_dump_json(by_alias=True)
        properties = json.loads(properties)

        item = pystac.Item(
            id=item_name.strip().split(".")[0],
            geometry=self.item_geometry_model(**ncattrs_to_geometry(item_data)).model_dump(),
            bbox=ncattrs_to_bbox(item_data),
            properties=properties,
            datetime=None,
        )

        # Add datacube extension
        try:
            dc_helper = DataCubeHelper(item_data, st, ed)
            dc_ext = DatacubeExtension.ext(item, add_if_missing=True)
            dc_ext.apply(dimensions=dc_helper.dimensions, variables=dc_helper.variables)
        except Exception as e:
            raise Exception("Failed to add Datacube extension") from e

        # Add THREDDS extension
        try:
            thredds_helper = THREDDSHelper(item_data["access_urls"])
            thredds_ext = THREDDSExtension.ext(item)
            thredds_ext.apply(thredds_helper.services, [])
        except Exception as e:
            raise Exception("Failed to add THREDDS extension") from e

        # Add Marble network extension
        item.properties["marble:host_node"] = "UofTRedOak"
        item.properties["marble:is_local"] = True
        item.stac_extensions.append(
            "https://raw.githubusercontent.com/DACCS-Climate/marble-stac-extension/v1.0.0/json-schema/schema.json"
        )

        try:
            item.validate()
        except STACValidationError:
            raise Exception("Failed to validate STAC item") from e

        return json.loads(json.dumps(item.to_dict()))
        # return item


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NEX GDDP CMIP6 STAC populator from a THREDDS catalog or NCML XML.")
    parser.add_argument("stac_host", type=str, help="STAC API address")
    parser.add_argument("href", type=str, help="URL to a THREDDS catalog or a NCML XML with NEX GDDP metadata.")
    parser.add_argument("--update", action="store_true", help="Update collection and its items")
    parser.add_argument(
        "--mode",
        choices=["full", "single"],
        default="full",
        help="Operation mode, processing the full dataset or only the single reference.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help=(
            "Override configuration file for the populator. "
            "By default, uses the adjacent configuration to the implementation class."
        ),
    )
    parser.add_argument(
        "--add-magpie-item-links",
        action="store_true",
        help="If specified, the item level resource links are added to Magpie.",
    )
    add_request_options(parser)
    return parser


def runner(ns: argparse.Namespace) -> Optional[int] | NoReturn:
    LOGGER.info(f"Arguments to call: {vars(ns)}")

    with Session() as session:
        apply_request_options(session, ns)
        if ns.mode == "full":
            data_loader = THREDDSLoader(ns.href, session=session)
        else:
            # To be implemented
            data_loader = ErrorLoader()

        c = NEXGDDPCMIP6populator(
            ns.stac_host,
            ns.href,
            data_loader,
            update=ns.update,
            session=session,
            config_file=ns.config,
            log_debug=ns.debug,
            add_magpie_item_links=ns.add_magpie_item_links,
        )
        c.ingest()


def main(*args: str) -> Optional[int]:
    parser = make_parser()
    ns = parser.parse_args(args or None)
    return runner(ns)


if __name__ == "__main__":
    main()
