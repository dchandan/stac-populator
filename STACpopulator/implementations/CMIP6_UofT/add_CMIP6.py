import argparse
import json
import logging
import os
from typing import Any, MutableMapping, NoReturn, Optional, Tuple, Union
from urllib.parse import urlparse

import cftime
import netCDF4 as nc
from marble_client import MarbleClient
from pystac import STACValidationError
from pystac.extensions.datacube import DatacubeExtension
from requests.sessions import Session

from STACpopulator.cli import add_request_options, apply_request_options
from STACpopulator.extensions.cmip6 import CMIP6Helper, CMIP6Properties
from STACpopulator.extensions.datacube import DataCubeHelper
from STACpopulator.extensions.thredds import THREDDSExtension, THREDDSHelper
from STACpopulator.input import ErrorLoader, GenericLoader, THREDDSLoader
from STACpopulator.models import GeoJSONPolygon
from STACpopulator.populator_base import STACpopulatorBase

LOGGER = logging.getLogger(__name__)


class CMIP6populator(STACpopulatorBase):
    item_properties_model = CMIP6Properties
    item_geometry_model = GeoJSONPolygon

    def __init__(
        self,
        stac_host: str,
        data_loader: GenericLoader,
        catalog_url: str,
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
        self, item_name: str, item_data: MutableMapping[str, Any], item_loc: str
    ) -> Union[None, MutableMapping[str, Any]]:
        """Creates the STAC item.

        :param item_name: name of the STAC item. Interpretation of name is left to the input loader implementation
        :type item_name: str
        :param item_data: dictionary like representation of all information on the item
        :type item_data: MutableMapping[str, Any]
        :return: _description_
        :rtype: MutableMapping[str, Any]
        """
        # Add CMIP6 extension
        try:
            # The "data version" of CMIP6 data is not included as a metadata in the netCDF file itself.
            # However, the version is used in the Data Reference Syntax system used to hierarchically organize the data
            # and can be therefore extracted from the full path to the data file. Here, I am extracting that information
            # and including it in the attributes that are generated and returned by the THREDDS server. This way, the
            # version information can be easily passed to the Pydantic data model for the CMIP6 data, used within the
            # CMIP6Helper class.
            item_data["attributes"]["version"] = item_loc.strip().split("/")[-2]

            # Get the item's start and end date from the item name
            st, ed, _ = self.get_item_temporal_information(item_data["access_urls"]["OPENDAP"])

            cmip_helper = CMIP6Helper(item_data, self.item_geometry_model, st, ed)
            item = cmip_helper.stac_item()
        except Exception as e:
            raise Exception("Failed to add CMIP6 extension") from e

        # Add datacube extension
        try:
            dc_helper = DataCubeHelper(item_data)
            dc_ext = DatacubeExtension.ext(item, add_if_missing=True)
            dc_ext.apply(dimensions=dc_helper.dimensions, variables=dc_helper.variables)
        except Exception as e:
            raise Exception("Failed to add Datacube extension") from e

        try:
            thredds_helper = THREDDSHelper(item_data["access_urls"])
            thredds_ext = THREDDSExtension.ext(item)
            thredds_links = thredds_helper.links if self.add_magpie_item_links else []
            thredds_ext.apply(thredds_helper.services, thredds_links)
        except Exception as e:
            raise Exception("Failed to add THREDDS extension") from e

        # Add Marble network extension
        item.properties["marble:host_node"] = self._marble_host_node
        item.properties["marble:is_local"] = True
        item.stac_extensions.append(
            "https://raw.githubusercontent.com/DACCS-Climate/marble-stac-extension/v1.0.0/json-schema/schema.json"
        )

        try:
            item.validate()
        except STACValidationError:
            raise Exception("Failed to validate STAC item") from e

        return json.loads(json.dumps(item.to_dict()))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CMIP6 STAC populator from a THREDDS catalog or NCML XML.")
    parser.add_argument("stac_host", type=str, help="STAC API address")
    parser.add_argument("href", type=str, help="URL to a THREDDS catalog or a NCML XML with CMIP6 metadata.")
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
    parser.add_argument("--add-magpie-item-links", action="store_true")
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

        c = CMIP6populator(
            ns.stac_host,
            data_loader,
            ns.href,
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
