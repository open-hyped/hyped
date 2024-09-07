import os

import datasets
import pytest

import hyped.io.datasets  # noqa: F401


class TestCasDataset:
    @pytest.fixture
    def data_dir(self):
        return "./tests/artifacts/cas/"

    def test_load_data(self, data_dir, tmpdir):
        # load dataset
        ds = datasets.load_dataset(
            "hyped.io.datasets.cas",
            typesystem=os.path.join(data_dir, "typesystem.xml"),
            data_files={"train": os.path.join(data_dir, "cas.*")},
            cache_dir=os.path.join(tmpdir, "cache"),
        )

        # check dataset length
        assert len(ds["train"]) == 2
        # check features
        assert "sofa" in ds["train"].features
        # label features
        assert "cassis.Label:label" in ds["train"].features
        # entity features
        assert "cassis.Entity:begin" in ds["train"].features
        assert "cassis.Entity:end" in ds["train"].features
        assert "cassis.Entity:entityType" in ds["train"].features
        # relation features
        assert "cassis.Relation:source" in ds["train"].features
        assert "cassis.Relation:target" in ds["train"].features

        # check annotations
        for example in ds["train"]:
            text = example["sofa"]
            # test label annotation
            assert example["cassis.Label:label"] == ["Document"]
            # test entity annotation features
            assert len(example["cassis.Entity:entityType"]) == 2
            assert len(example["cassis.Entity:entityType"]) == len(example["cassis.Entity:begin"])
            assert len(example["cassis.Entity:entityType"]) == len(example["cassis.Entity:end"])
            # test relation annotation features
            assert len(example["cassis.Relation:source"]) == 1
            assert len(example["cassis.Relation:source"]) == len(example["cassis.Relation:target"])

            # test entity content
            for eType, begin, end in zip(
                example["cassis.Entity:entityType"],
                example["cassis.Entity:begin"],
                example["cassis.Entity:end"],
            ):
                assert eType in {"ORG", "LOC"}
                # test content
                if eType == "ORG":
                    assert text[begin:end] == "U.N."
                if eType == "LOC":
                    assert text[begin:end] == "Baghdad"

            # test relation content
            for src, tgt in zip(
                example["cassis.Relation:source"],
                example["cassis.Relation:target"],
            ):
                assert example["cassis.Entity:entityType"][src] == "ORG"
                assert example["cassis.Entity:entityType"][tgt] == "LOC"

    def test_load_specific_types_only(self, data_dir, tmpdir):
        # load dataset
        ds = datasets.load_dataset(
            "hyped.io.datasets.cas",
            typesystem=os.path.join(data_dir, "typesystem.xml"),
            data_files={"train": os.path.join(data_dir, "cas.*")},
            types=["cassis.Label"],
            cache_dir=os.path.join(tmpdir, "cache"),
        )

        # check dataset length
        assert len(ds["train"]) == 2
        assert "sofa" in ds["train"].features
        # label should be included
        assert "cassis.Label:label" in ds["train"].features
        # entity should be excluded
        assert "cassis.Entity:begin" not in ds["train"].features
        assert "cassis.Entity:end" not in ds["train"].features
        assert "cassis.Entity:entityType" not in ds["train"].features
        # relation should be excluded
        assert "cassis.Relation:source" not in ds["train"].features
        assert "cassis.Relation:target" not in ds["train"].features

    def test_error_on_required_type(self, data_dir, tmpdir):
        with pytest.raises(RuntimeError):
            # load dataset
            datasets.load_dataset(
                "hyped.io.datasets.cas",
                typesystem=os.path.join(data_dir, "typesystem.xml"),
                data_files={"train": os.path.join(data_dir, "cas.*")},
                types=[
                    "cassis.Label",
                    "cassis.Relation",  # relation require entities
                ],
                cache_dir=os.path.join(tmpdir, "cache"),
            )
