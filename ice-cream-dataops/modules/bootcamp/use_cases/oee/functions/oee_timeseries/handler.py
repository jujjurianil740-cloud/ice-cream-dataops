from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import islice
from typing import Any, Dict

from cognite.client import CogniteClient
from cognite.client.data_classes.data_modeling import NodeId, ViewId
from cognite.client.data_classes.data_modeling.cdm.v1 import CogniteAsset, CogniteTimeSeries, CogniteTimeSeriesApply
from cognite.client.data_classes.filters import Prefix, ContainsAny
from cognite.client.exceptions import CogniteNotFoundError

import numpy as np

from cognite.client.config import global_config
global_config.disable_pypi_version_check = True


ALL_SITES = [
    "Houston",
    "Oslo",
    "Kuala_Lumpur",
    "Hannover",
    "Nuremberg",
    "Marseille",
    "Sao_Paulo",
    "Chicago",
    "Rotterdam",
    "London",
]

OEE_SPACE = "oee_ts_space"
SOURCE_SPACE = "icapi_dm_space"


def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def to_datetime(value) -> datetime | None:
    """Return a timezone-aware datetime from a datetime or a millisecond epoch value."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    return None


def column_name(column) -> str:
    """Convert a dataframe column label (NodeId or string) to a plain external ID string."""
    if isinstance(column, NodeId):
        return column.external_id
    return str(column)


def backfill_leading_gaps(df, col):
    """Fill leading NaNs with the opposite of the first valid value. Safe when the column is empty."""
    first_idx = df[col].first_valid_index()
    if first_idx is None:
        return False
    first_valid_value = df[col].loc[first_idx]
    df[col] = df[col].fillna(value=1.0 if first_valid_value == 0.0 else 0.0)
    return True


def get_time_series_for_site(client: CogniteClient, site, space):
    this_site = site.lower()
    sub_tree_root = client.data_modeling.instances.retrieve_nodes(
        NodeId(space, this_site),
        node_cls=CogniteAsset
    )

    if not sub_tree_root:
        print(
            f"----No CogniteAssets in CDF for {site}!----\n"
            f"    Run the 'Create Cognite Asset Hierarchy' transformation!"
        )
        return

    sub_tree_nodes = client.data_modeling.instances.list(
        instance_type=CogniteAsset,
        filter=Prefix(property=["cdf_cdm", "CogniteAsset/v1", "path"], value=sub_tree_root.path),
        limit=None
    )

    if not sub_tree_nodes:
        print(
            f"----No CogniteTimeSeries in CDF for {site}!----\n"
            f"    Run the 'Contextualize Timeseries and Assets' transformation!"
        )
        return

    value_list = [{"space": node.space, "externalId": node.external_id} for node in sub_tree_nodes]

    time_series = [
        client.data_modeling.instances.search(
            view=ViewId("cdf_cdm", "CogniteTimeSeries", "v1"),
            instance_type=CogniteTimeSeries,
            space=space,
            filter=ContainsAny(property=["cdf_cdm", "CogniteTimeSeries/v1", "assets"], values=batch),
            limit=None
        )
        for batch in batcher(value_list, 20)
    ]

    # Combine list of batch results into a single NodeList
    time_series = [node for nodelist in time_series for node in nodelist]

    if not time_series:
        print("No CogniteTimeSeries in the CogniteCore Data Model (cdf_cdm Space)")

    return time_series


def handle(client: CogniteClient, data: Dict[str, Any] = None) -> None:
    data = data or {}

    # Kept as a timedelta so it can be subtracted from datetime values
    lookback = timedelta(minutes=data.get("lookback_minutes", 60))
    sites = data.get("sites") or ALL_SITES

    print(f"Processing datapoints for these sites: {sites}")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_site, client, lookback, site) for site in sites]
        for f in futures:
            f.result()


def process_site(client, lookback: timedelta, site):
    timeseries = get_time_series_for_site(client, site, SOURCE_SPACE)
    if not timeseries:
        return

    asset_eids = list(set([item.external_id.split(sep=":")[0] for item in timeseries]))
    instance_ids = [NodeId(space=SOURCE_SPACE, external_id=ts.external_id) for ts in timeseries]
    all_latest_dps = client.time_series.data.retrieve_latest(instance_id=instance_ids)

    # Group latest datapoints by equipment. Match on the exact asset prefix, not a substring.
    assets_dps = {
        asset: [dp for dp in all_latest_dps if dp.instance_id.external_id.split(":")[0] == asset]
        for asset in asset_eids
    }

    for asset, latest_dps in assets_dps.items():
        print(f"Calculating OEE for {asset}")

        count_col = f"{asset}:count"
        good_col = f"{asset}:good"
        status_col = f"{asset}:status"
        planned_status_col = f"{asset}:planned_status"

        # Latest timestamp across this asset's series, normalized to datetime
        timestamps = [to_datetime(dp.timestamp) for dp in latest_dps if dp.timestamp]
        timestamps = [ts for ts in timestamps if ts is not None]
        if not timestamps:
            print(f"{asset}: no latest timestamps found, skipping")
            continue
        end = min(timestamps)

        dps_df = client.time_series.data.retrieve_dataframe(
            instance_id=[dp.instance_id for dp in latest_dps],
            start=end - lookback,
            end=end,
            aggregates=["sum"],
            granularity="1m",
            include_aggregate_name=False,
            limit=None
        )

        # Convert NodeId column labels to plain external IDs, e.g. "HOPRAGTA812:count"
        dps_df.columns = [column_name(c) for c in dps_df.columns]

        required = [count_col, good_col, status_col, planned_status_col]
        missing = [col for col in required if col not in dps_df.columns]
        if missing:
            print(f"{asset}: missing required columns {missing}, skipping")
            continue

        # Frontfill because "planned_status" and "status" only have datapoints when the value changes
        dps_df = dps_df.ffill()

        # Fill leading gaps with the opposite of the first valid value
        if not backfill_leading_gaps(dps_df, planned_status_col):
            print(f"{asset}: no planned_status datapoints in range, skipping")
            continue
        if not backfill_leading_gaps(dps_df, status_col):
            print(f"{asset}: no status datapoints in range, skipping")
            continue

        count_dps = dps_df[count_col]
        good_dps = dps_df[good_col]
        status_dps = dps_df[status_col]
        planned_status_dps = dps_df[planned_status_col]

        # Calculate the components of OEE
        dps_df[f"{asset}:off_spec"] = count_dps - good_dps
        dps_df[f"{asset}:quality"] = good_dps / count_dps
        dps_df[f"{asset}:performance"] = (count_dps / status_dps) / (60.0 / 3.0)
        dps_df[f"{asset}:availability"] = status_dps / planned_status_dps
        dps_df[f"{asset}:oee"] = (
            dps_df[f"{asset}:quality"]
            * dps_df[f"{asset}:performance"]
            * dps_df[f"{asset}:availability"]
        )

        # Fill in the divide by zeros
        dps_df = dps_df.fillna(value=0.0)
        dps_df = dps_df.replace([np.inf, -np.inf], 0.0)

        # Drop input series, keep only the calculated OEE series
        dps_df = dps_df.drop(columns=[count_col, good_col, status_col, planned_status_col])

        to_insert = [
            {
                "instance_id": NodeId(space=OEE_SPACE, external_id=external_id),
                "datapoints": list(zip(dps_df[external_id].index, dps_df[external_id]))
            }
            for external_id in dps_df.columns
        ]

        try:
            client.time_series.data.insert_multiple(to_insert)
        except CogniteNotFoundError as e:
            # Create the missing OEE timeseries since they don't exist
            ts_to_create = []
            for node_id in e.not_found:
                print(f"Creating CogniteTimeSeries {node_id}")

                external_id = node_id["instanceId"]["externalId"]

                # Change external_id to a readable name
                # Ex: "OSLPROFILTRASYS185:off_spec" to "OSLPROFILTRASYS185 Off Spec"
                name = external_id.split(":")
                name[-1] = name[-1].replace("_", " ").title()

                ts_to_create.append(
                    CogniteTimeSeriesApply(
                        space=OEE_SPACE,
                        external_id=external_id,
                        name=" ".join(name),
                        is_step=False,
                        time_series_type="numeric",
                    )
                )

            client.data_modeling.instances.apply(ts_to_create)
            client.time_series.data.insert_multiple(to_insert)