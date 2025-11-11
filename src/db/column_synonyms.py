COLUMN_SYNONYMS = {
    "item_no": [
        "item no", "item number", "product code", "sku", "product id", "model number",
        "item id", "article number", "product identifier", "stock keeping unit"
    ],

    "image_url": [
        "image", "image url", "photo", "picture link", "thumbnail", "product image"
    ],

    "date": [
        "date", "created date", "entry date", "transaction date", "record date", "updated date"
    ],

    "customer_name": [
         "customer", "customer name", "retailer", "client name", "buyer", "distributor"
    ],

    "company": [
         "company"
    ],

    "group_name": [
        "group", "group name", "product group", "category group", "item group", "brand group"
    ],

    "jgroup": [
        "jgroup", "jewelry group", "jewellery group", "jewel group", "ornament group", "jewelry category"
    ],

    "retail_range": [
        "retail range", "retail price range", "price bracket", "price range", "retail band", "pricing range"
    ],

    "range_name": [
        "range", "price range name"
    ],

    "main_category": [
        "main category", "primary category", "top level category", "main division"
    ],

    "subcategory": [
        "subcategory", "sub category", "child category", "secondary category"
    ],

    "collection": [
        "collection", "series", "theme", "product collection"
    ],

    "division": [
        "division", "department", "business division", "section"
    ],

    "diamond_ctw_fraction": [
        "diamond ctw fraction", "ctw fraction", "carat fraction", "diamond weight fraction"
    ],

    "custom_sd_ctrshap": [
        "custom sd ctrshap", "center shape", "diamond shape", "custom shape", "central stone shape"
    ],

    "sdc_mis_item_status": [
        "sdc mis item status", "item status", "status", "product status", "availability status"
    ],

    "new_tag": [
        "new tag", "is new", "new item", "recent", "latest", "tag status"
    ],

    "diamond_ctw_range": [
        "diamond ctw range", "ctw range", "diamond weight range", "carat range", "stone weight range"
    ],

    "custom_sd_ctrdesc": [
        "custom sd ctrdesc", "center description", "diamond description", "stone description", "center stone desc"
    ],

    "secondary_sales_qty": [
        "secondary sales qty", "secondary quantity", "sales quantity", "sold quantity", "qty sold"
    ],

    "secondary_sales_total_cost": [
        "secondary sales total cost", "sales total cost", "total sales cost", "secondary total cost", "sales cost"
    ],

    "secondary_sales_value": [
        "secondary sales value", "sales value", "total sales value", "secondary value", "revenue"
    ],

    "inventory_qty_final": [
        "inventory qty final", "final inventory quantity", "inventory quantity", "stock quantity", "available quantity"
    ],

    "inventory_cost_final": [
        "inventory cost final", "final inventory cost", "inventory cost", "stock cost", "total cost"
    ],

    "open_memo_qty": [
        "open memo qty", "memo quantity", "quantity on memo", "memo stock quantity"
    ],

    "open_memo_amount": [
        "open memo amount", "memo amount", "memo value", "memo total value", "total memo amount"
    ],

    "open_order_qty_asset": [
        "open order qty asset", "open asset order quantity", "asset order qty", "asset order quantity"
    ],

    "open_order_amount_asset": [
        "open order amount asset", "asset order amount", "asset order value", "open asset value"
    ],

    "open_order_qty_memo": [
        "open order qty memo", "memo order quantity", "open memo order qty", "memo order qty"
    ],

    "open_order_amount_memo": [
        "open order amount memo", "memo order amount", "open memo order value", "memo order value"
    ],
}


COLUMN_ALIAS_OVERRIDE = {
    "open order amount memo": "open_order_amount_memo",
    "open memo order value": "open_order_amount_memo",
    "memo order amount": "open_order_amount_memo",
    "memo order value": "open_order_amount_memo",
}


def normalize_column_name(col: str) -> str:
    """Normalize column variations safely."""
    return (
        col.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .strip()
    )
