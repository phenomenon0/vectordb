from __future__ import annotations


def test_create_and_drop_collection(client, collection_name):
    assert client.collection_exists(collection_name) is False
    client.create_collection(collection_name, dim=16)
    assert client.collection_exists(collection_name) is True
    client.delete_collection(collection_name)
    assert client.collection_exists(collection_name) is False


def test_drop_missing_collection_is_safe(client, collection_name):
    client.delete_collection(collection_name)
    assert client.collection_exists(collection_name) is False
