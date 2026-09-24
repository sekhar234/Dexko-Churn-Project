"""Source-table contracts extracted from the latest churn implementation.

These are the EDW-style source columns needed by the current snapshot/load/feature
path. The local loader will normalize them to code-friendly names later.
"""

SOURCE_CONTRACTS = {
    "dimcustomer": [
        "Harmonized Sold To Customer Name", "Customer Key", "Customer Name",
        "Customer Group", "Customer Group Name", "Credit Max", "Customer Id",
        "Sales Rep", "Data Area Id", "Intercompany Flag", "Region",
        "Reporting Segment", "InventSiteId",
    ],
    "factsalesinvoice": [
        "InvoiceDate", "NetAmountExtended", "Quantity", "DataAreaId",
        "SalesOrderCreatedDate", "Warehouse Location Key", "Product Key",
        "InvoiceAccountCustomerKey", "CustomerShipTokey", "ShiptoSeq",
        "InvoiceId", "City", "State", "ZipCode", "Country", "Street",
        "SalesId", "SalesOrderDate", "InterCompanyPosted",
    ],
    "dimproduct": [
        "Product Key", "MGR L1", "MGR L2", "Product Name", "Item Id",
        "Primary Vendor Group", "Item Group Id", "Item Group Name",
        "Product Description", "Product Type",
    ],
    "dimwarehouselocation": [
        "Warehouse Location Key", "BU Level 1", "BU Level 2", "BU Level 3",
        "InventSiteId", "Invent Location Id", "Business Unit", "InventSideIdName",
    ],
    "dimcustomershipto": [
        "CustomerShipTokey", "CustomerName", "City", "State", "ZipCode",
        "Country", "Territory", "SalesRep", "Region", "RegionalManager", "VP",
        "ServicingBranch", "ReportingSegment",
    ],
}

RENAME_MAP = {
    "Harmonized Sold To Customer Name": "harmonizedsoldtocustomername",
    "BU Level 1": "bulevel1",
    "Customer Group Name": "customergroupname",
    "Credit Max": "creditmax",
    "Customer Name": "customername",
    "Customer Group": "customergroup",
    "Customer Key": "customerkey",
    "Customer Id": "customerid",
    "Sales Rep": "salesrep",
    "Data Area Id": "dataareaid",
    "Intercompany Flag": "intercompanyflag",
    "Region": "region",
    "Reporting Segment": "reportingsegment",
    "InvoiceDate": "invoicedate",
    "NetAmountExtended": "netamountextended",
    "Quantity": "quantity",
    "DataAreaId": "dataareaid",
    "SalesOrderCreatedDate": "salesordercreateddate",
    "Warehouse Location Key": "warehouselocationkey",
    "Product Key": "productkey",
    "InvoiceAccountCustomerKey": "invoiceaccountcustomerkey",
    "CustomerShipTokey": "customershiptokey",
    "ShiptoSeq": "shiptoseq",
    "InvoiceId": "invoiceid",
    "City": "city",
    "State": "state",
    "ZipCode": "zipcode",
    "Street": "street",
    "SalesId": "salesid",
    "SalesOrderDate": "salesorderdate",
    "InterCompanyPosted": "intercompanyflag_inv",
    "MGR L1": "mgrl1",
    "MGR L2": "mgrl2",
    "Product Name": "productname",
    "Item Id": "itemid",
    "Primary Vendor Group": "primaryvendorgroup",
    "Item Group Id": "itemgroupid",
    "Item Group Name": "itemgroupname",
    "Product Description": "productdescription",
    "Product Type": "producttype",
    "BU Level 2": "bulevel2",
    "BU Level 3": "bulevel3",
    "Invent Location Id": "inventlocationid",
    "Business Unit": "businessunit",
}
