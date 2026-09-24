from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.synthetic.generator import load_config, generate
from src.synthetic.qa import run_qa

@pytest.fixture(scope="session")
def generated_bundle():
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root/"config"/"synthetic.yml")
    cfg["scales"]["test"]={"customers":60,"products":36,"warehouses":6}
    cfg["scale"]="test"; cfg["seed"]=42
    return generate(cfg)


def test_generator_contracts(generated_bundle):
    b=generated_bundle
    assert len(b.factsalesinvoice)>1000
    assert {"InvoiceDate","NetAmountExtended","Quantity","InvoiceAccountCustomerKey","Product Key","ZipCode"}.issubset(b.factsalesinvoice.columns)
    assert {"OEM","Stocking Dealer"}.issubset(set(b.dimcustomer["Customer Group Name"]))


def test_qa_passes(generated_bundle):
    qa=run_qa(generated_bundle)
    assert all(qa["checks"].values())
