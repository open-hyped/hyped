from typing import Generic, TypeVar
from unittest.mock import patch

import datasets

from hyped.core.features.dtypes import Float32Type, Int64Type, MappingType, StringType
from hyped.core.flow import DataFlow
from hyped.core.module import DataFlowModule
from hyped.typing import Float32, Int64, String


class TestDataFlowModule:
    def test_build_src_dtype(self):
        class ModuleA(DataFlowModule):
            def call(self, x: Int64):
                ...

        class ModuleB(DataFlowModule):
            def call(self, x: Int64, y: String, z: Float32):
                ...

        T = TypeVar("T")

        class ModuleC(DataFlowModule, Generic[T]):
            def call(self, x: T):
                ...

        assert ModuleA()._build_src_dtype() == MappingType.construct({"x": Int64Type})
        assert ModuleB()._build_src_dtype() == MappingType.construct(
            {"x": Int64Type, "y": StringType, "z": Float32Type}
        )
        assert ModuleC[String]()._build_src_dtype() == MappingType.construct({"x": StringType})

    def test_build_flow(self):
        ds = datasets.Dataset.from_dict({"x": [1, 2, 3], "y": [3, 4, 5]})

        class Module(DataFlowModule):
            def call(self, x: Int64, y: Int64):
                return x + y

        flow = Module().flow
        assert flow.collect_feature.dtype == MappingType.construct({"output": Int64Type})
        assert flow.apply(ds).to_dict() == {"output": [4, 6, 8]}

        class Module(DataFlowModule):
            def call(self, x: Int64, y: Int64):
                return {"sum": x + y}

        flow = Module().flow
        assert flow.collect_feature.dtype == MappingType.construct({"sum": Int64Type})
        assert flow.apply(ds).to_dict() == {"sum": [4, 6, 8]}

    def test_flow_property(self):
        flow = DataFlow(datasets.Features({"x": datasets.Value("int64")}))
        in_graph = [None]

        class Module(DataFlowModule):
            def call(self, x: Int64):
                assert isinstance(self.flow, DataFlow)
                in_graph[0] = self.flow._graph
                return x

        # call the module and check that the property
        # references the correct graph
        Module().call(x=flow.source["x"])
        assert flow._graph == in_graph[0]

        # check that the flow is set when building directly
        _ = Module().flow

    @patch("hyped.core.module.plot_data_flow")
    def test_plot(self, mock_plot_data_flow):
        class Module(DataFlowModule):
            def call(self, x: Int64, y: Int64):
                return x + y

        Module().plot()
        mock_plot_data_flow.assert_called_once()
