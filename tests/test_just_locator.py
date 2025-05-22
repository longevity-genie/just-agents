"""
Unit tests for the JustLocator implementation.
"""

import unittest
import gc
from typing import List, Optional
import sys
import os

# Add the core module to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from just_agents.just_locator import JustLocator, EntityIdentifier


class MockEntity:
    """Mock entity class with name attribute."""
    
    def __init__(self, name: str, value: int = 0):
        self.name = name
        self.value = value
    
    def __repr__(self):
        return f"MockEntity(name='{self.name}', value={self.value})"
    
    def __eq__(self, other):
        if not isinstance(other, MockEntity):
            return False
        return self.name == other.name and self.value == other.value


class MockEntityNoName:
    """Mock entity class without name attribute."""
    
    def __init__(self, data: str):
        self.data = data
    
    def __repr__(self):
        return f"MockEntityNoName(data='{self.data}')"


class MockEntityWithTitle:
    """Mock entity class with title attribute."""
    
    def __init__(self, title: str, description: str):
        self.title = title
        self.description = description
    
    def __repr__(self):
        return f"MockEntityWithTitle(title='{self.title}', description='{self.description}')"


class TestJustLocator(unittest.TestCase):
    """Test cases for JustLocator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.locator = JustLocator[MockEntity](entity_config_identifier_attr="name")
        
    def test_publish_entity_basic(self):
        """Test basic entity publishing."""
        entity = MockEntity("test", 42)
        codename = self.locator.publish_entity(entity)
        
        self.assertIsInstance(codename, str)
        self.assertGreater(len(codename), 0)
        
        # Verify entity can be retrieved
        retrieved = self.locator.get_entity_by_codename(codename)
        self.assertIs(retrieved, entity)
        
    def test_publish_same_entity_twice(self):
        """Test that publishing the same entity twice returns the same codename."""
        entity = MockEntity("test", 42)
        codename1 = self.locator.publish_entity(entity)
        codename2 = self.locator.publish_entity(entity)
        
        self.assertEqual(codename1, codename2)
        
    def test_get_entity_codename(self):
        """Test getting codename for an entity instance."""
        entity = MockEntity("test", 42)
        codename = self.locator.publish_entity(entity)
        
        retrieved_codename = self.locator.get_entity_codename(entity)
        self.assertEqual(retrieved_codename, codename)
        
    def test_get_entity_codename_unregistered(self):
        """Test getting codename for unregistered entity."""
        entity = MockEntity("test", 42)
        codename = self.locator.get_entity_codename(entity)
        self.assertIsNone(codename)
        
    def test_get_entities_by_config_identifier(self):
        """Test getting entities by config identifier."""
        entity1 = MockEntity("alice", 1)
        entity2 = MockEntity("bob", 2)
        entity3 = MockEntity("alice", 3)
        
        self.locator.publish_entity(entity1)
        self.locator.publish_entity(entity2)
        self.locator.publish_entity(entity3)
        
        alices = self.locator.get_entities_by_config_identifier("alice")
        self.assertEqual(len(alices), 2)
        self.assertIn(entity1, alices)
        self.assertIn(entity3, alices)
        
        bobs = self.locator.get_entities_by_config_identifier("bob")
        self.assertEqual(len(bobs), 1)
        self.assertIn(entity2, bobs)
        
    def test_get_entity_codenames_by_config_identifier(self):
        """Test getting codenames by config identifier."""
        entity1 = MockEntity("alice", 1)
        entity2 = MockEntity("alice", 2)
        
        codename1 = self.locator.publish_entity(entity1)
        codename2 = self.locator.publish_entity(entity2)
        
        codenames = self.locator.get_entity_codenames_by_config_identifier("alice")
        self.assertEqual(len(codenames), 2)
        self.assertIn(codename1, codenames)
        self.assertIn(codename2, codenames)
        
    def test_get_entity_codenames_by_class(self):
        """Test getting codenames by class."""
        entity1 = MockEntity("test1", 1)
        entity2 = MockEntity("test2", 2)
        
        codename1 = self.locator.publish_entity(entity1)
        codename2 = self.locator.publish_entity(entity2)
        
        codenames = self.locator.get_entity_codenames_by_class(MockEntity)
        self.assertEqual(len(codenames), 2)
        self.assertIn(codename1, codenames)
        self.assertIn(codename2, codenames)
        
    def test_arbitrary_search(self):
        """Test arbitrary search functionality."""
        entity1 = MockEntity("test1", 10)
        entity2 = MockEntity("test2", 5)
        entity3 = MockEntity("test3", 15)
        
        self.locator.publish_entity(entity1)
        self.locator.publish_entity(entity2)
        self.locator.publish_entity(entity3)
        
        # Find entities with value > 7
        results = self.locator.arbitrary_search(MockEntity, lambda e: e.value > 7)
        self.assertEqual(len(results), 2)
        self.assertIn(entity1, results)
        self.assertIn(entity3, results)
        
    def test_unpublish_entity(self):
        """Test unpublishing an entity."""
        entity = MockEntity("test", 42)
        codename = self.locator.publish_entity(entity)
        
        # Verify entity exists
        retrieved = self.locator.get_entity_by_codename(codename)
        self.assertIs(retrieved, entity)
        
        # Unpublish
        result = self.locator.unpublish_entity(entity)
        self.assertTrue(result)
        
        # Verify entity is gone
        retrieved = self.locator.get_entity_by_codename(codename)
        self.assertIsNone(retrieved)
        
    def test_unpublish_entity_by_codename(self):
        """Test unpublishing by codename."""
        entity = MockEntity("test", 42)
        codename = self.locator.publish_entity(entity)
        
        result = self.locator.unpublish_entity_by_codename(codename)
        self.assertTrue(result)
        
        retrieved = self.locator.get_entity_by_codename(codename)
        self.assertIsNone(retrieved)
        
    def test_unpublish_entities_by_config_identifier(self):
        """Test unpublishing by config identifier."""
        entity1 = MockEntity("alice", 1)
        entity2 = MockEntity("alice", 2)
        entity3 = MockEntity("bob", 3)
        
        self.locator.publish_entity(entity1)
        self.locator.publish_entity(entity2)
        self.locator.publish_entity(entity3)
        
        removed_count = self.locator.unpublish_entities_by_config_identifier("alice")
        self.assertEqual(removed_count, 2)
        
        # Verify alice entities are gone
        alices = self.locator.get_entities_by_config_identifier("alice")
        self.assertEqual(len(alices), 0)
        
        # Verify bob entity still exists
        bobs = self.locator.get_entities_by_config_identifier("bob")
        self.assertEqual(len(bobs), 1)
        
    def test_get_identifier_by_instance(self):
        """Test getting identifier by entity instance."""
        entity = MockEntity("test", 42)
        codename = self.locator.publish_entity(entity)
        
        identifier = self.locator.get_identifier_by_instance(entity)
        self.assertIsNotNone(identifier)
        self.assertEqual(identifier.entity_codename, codename)
        self.assertEqual(identifier.entity_class, MockEntity)
        self.assertEqual(identifier.entity_config_identifier, "test")


class TestEntityConfigIdentifierFallback(unittest.TestCase):
    """Test cases for entity config identifier fallback behavior."""
    
    def test_fallback_to_class_name(self):
        """Test fallback to class name when attribute doesn't exist."""
        locator = JustLocator[MockEntityNoName]()  # Uses default 'name' attr
        
        entity = MockEntityNoName("test_data")
        codename = locator.publish_entity(entity)
        
        # Should be able to find by class name since 'name' attribute doesn't exist
        entities = locator.get_entities_by_config_identifier("MockEntityNoName")
        self.assertEqual(len(entities), 1)
        self.assertIs(entities[0], entity)
        
    def test_custom_attribute_name(self):
        """Test using a custom attribute name for config identifier."""
        locator = JustLocator[MockEntityWithTitle](entity_config_identifier_attr="title")
        
        entity1 = MockEntityWithTitle("Product A", "Description A")
        entity2 = MockEntityWithTitle("Product B", "Description B")
        entity3 = MockEntityWithTitle("Product A", "Different description")
        
        locator.publish_entity(entity1)
        locator.publish_entity(entity2)
        locator.publish_entity(entity3)
        
        # Find by title
        product_a_entities = locator.get_entities_by_config_identifier("Product A")
        self.assertEqual(len(product_a_entities), 2)
        self.assertIn(entity1, product_a_entities)
        self.assertIn(entity3, product_a_entities)
        
        product_b_entities = locator.get_entities_by_config_identifier("Product B")
        self.assertEqual(len(product_b_entities), 1)
        self.assertIn(entity2, product_b_entities)


class TestEntityIdentifier(unittest.TestCase):
    """Test cases for EntityIdentifier."""
    
    def test_entity_identifier_creation(self):
        """Test creating an EntityIdentifier."""
        identifier = EntityIdentifier[MockEntity](
            entity_class=MockEntity,
            entity_codename="test-codename"
        )
        
        self.assertEqual(identifier.entity_class, MockEntity)
        self.assertEqual(identifier.entity_codename, "test-codename")
        
    def test_entity_config_identifier_property(self):
        """Test the entity_config_identifier property."""
        locator = JustLocator[MockEntity](entity_config_identifier_attr="name")
        entity = MockEntity("test_name", 42)
        
        codename = locator.publish_entity(entity)
        identifier = locator.get_identifier_by_instance(entity)
        
        self.assertEqual(identifier.entity_config_identifier, "test_name")
        
    def test_entity_config_identifier_without_locator(self):
        """Test that accessing entity_config_identifier without locator raises error."""
        identifier = EntityIdentifier[MockEntity](
            entity_class=MockEntity,
            entity_codename="test-codename"
        )
        
        with self.assertRaises(ValueError):
            _ = identifier.entity_config_identifier


class TestWeakReferences(unittest.TestCase):
    """Test cases for weak reference management."""
    
    def test_automatic_cleanup_on_garbage_collection(self):
        """Test that entities are automatically cleaned up when garbage collected."""
        locator = JustLocator[MockEntity](entity_config_identifier_attr="name")
        
        # Create and register an entity
        entity = MockEntity("test", 42)
        codename = locator.publish_entity(entity)
        
        # Verify entity exists
        retrieved = locator.get_entity_by_codename(codename)
        self.assertIs(retrieved, entity)
        
        # Remove our reference and force aggressive garbage collection
        del entity
        del retrieved  # Remove the retrieved reference too
        # Force multiple rounds of garbage collection
        for _ in range(3):
            gc.collect()
        
        # Entity should be automatically cleaned up
        retrieved = locator.get_entity_by_codename(codename)
        self.assertIsNone(retrieved)
        
        # Identifier should also be cleaned up
        self.assertNotIn(codename, locator._entity_codename_to_identifiers)


if __name__ == '__main__':
    unittest.main() 