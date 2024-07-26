import datasets
from datasets import Dataset, Features, Value

from hyped.data.flow import DataFlow
from hyped.data.flow.ops import collect

datasets.disable_caching()


input_features = Features(
    {
        "A": Value("int32"),
        "B": Value("int64"),
    }
)
input_data = {
    "A": [64],
    "B": [32],
}
ds = Dataset.from_dict(input_data, features=input_features)

flow = DataFlow(features=ds.features)

ds_out, _ = flow.apply(ds, collect=collect({"concat": flow.src_features.A + flow.src_features.B}))

print(ds_out[0])
