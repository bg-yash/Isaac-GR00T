from typing import List, Dict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DB_DIR = ROOT / "databases"
SKU_DB = DB_DIR / "skus.csv"
GRIPPER_DB = DB_DIR / "grippers.csv"


def get_gripper_db_collection():
    """
    Loads in the gripper database collection.
    """

    return pd.read_csv(GRIPPER_DB)


def get_gripper_ids() -> List:
    """
    Loads in the grippers and returns the gripper IDs as a list
    """
    return get_gripper_db_collection()["gripper_id"].tolist()


def get_sku_db_collection() -> Dict:
    """
    Loads in the SKU database collection.
    """

    return pd.read_csv(SKU_DB).set_index("product_id")["name"].to_dict()


def get_sku_id_by_product_name(sku_db: Dict, product_name: str) -> str | None:
    """
    Returns the SKU ID based on the product name.
    """
    sku_id = None
    for key, value in sku_db.items():
        if value == product_name:
            if sku_id is not None:
                print("Multiple SKUs found for product name. Returning first instance.")
            else:
                sku_id = key
    return sku_id
