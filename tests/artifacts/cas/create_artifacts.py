import os

import cassis
from cassis.typesystem import TYPE_NAME_STRING


def build_typesystem(path):
    # create sample typesystem
    typesystem = cassis.TypeSystem(add_document_annotation_type=False)
    # add label annotation
    label = typesystem.create_type(name="cassis.Label", supertypeName="uima.cas.TOP")
    typesystem.create_feature(domainType=label, name="label", rangeType=TYPE_NAME_STRING)
    # add entity annotation
    entity = typesystem.create_type(name="cassis.Entity")
    typesystem.create_feature(domainType=entity, name="entityType", rangeType=TYPE_NAME_STRING)
    # add bi-relation entity
    relation = typesystem.create_type(name="cassis.Relation", supertypeName="uima.cas.TOP")
    typesystem.create_feature(domainType=relation, name="source", rangeType=entity)
    typesystem.create_feature(domainType=relation, name="target", rangeType=entity)
    # save typesystem
    typesystem.to_xml(os.path.join(path, "typesystem.xml"))


def build_examples(path):
    # load test typesystem
    with open(os.path.join(path, "typesystem.xml"), "rb") as f:
        typesystem = cassis.load_typesystem(f)
    # get annotation types
    Entity = typesystem.get_type("cassis.Entity")
    Relation = typesystem.get_type("cassis.Relation")
    Label = typesystem.get_type("cassis.Label")
    # create cas object
    cas = cassis.Cas(typesystem=typesystem)
    cas.sofa_string = "U.N. official Ekeus heads for Baghdad."
    # create entities
    org = Entity(begin=0, end=4, entityType="ORG")
    loc = Entity(begin=30, end=37, entityType="LOC")
    # add annotations
    cas.add_all([org, loc, Relation(source=org, target=loc), Label(label="Document")])
    # save in json and xmi format
    cas.to_json(os.path.join(path, "cas.json"))
    cas.to_xmi(os.path.join(path, "cas.xmi"))


if __name__ == "__main__":
    build_typesystem(".")
    build_examples(".")
