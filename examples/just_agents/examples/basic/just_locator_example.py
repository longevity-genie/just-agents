"""
Example usage of the JustLocator with different entity types.
"""

from typing import List
from just_agents.just_locator import JustLocator, JustSingletonLocator


# Example entity classes
class User:
    def __init__(self, name: str, email: str):
        self.name = name  # This will be used as entity_config_identifier
        self.email = email
    
    def __repr__(self):
        return f"User(name='{self.name}', email='{self.email}')"


class Product:
    def __init__(self, title: str, price: float):
        self.title = title  # This will be used as entity_config_identifier  
        self.price = price
    
    def __repr__(self):
        return f"Product(title='{self.title}', price={self.price})"


class Service:
    def __init__(self, service_id: str, description: str):
        self.service_id = service_id
        self.description = description
    
    def __repr__(self):
        return f"Service(service_id='{self.service_id}', description='{self.description}')"


def example_usage():
    """Demonstrate various locator functionalities."""
    
    # Example 1: User locator using 'name' attribute
    user_locator = JustLocator[User](entity_config_identifier_attr="name")
    
    # Create and register users
    user1 = User("alice", "alice@example.com")
    user2 = User("bob", "bob@example.com")
    user3 = User("alice", "alice.jones@example.com")  # Another Alice
    
    codename1 = user_locator.publish_entity(user1)
    codename2 = user_locator.publish_entity(user2)
    codename3 = user_locator.publish_entity(user3)
    
    print(f"Registered users with codenames: {codename1}, {codename2}, {codename3}")
    
    # Find users by config identifier (name)
    alices = user_locator.get_entities_by_config_identifier("alice")
    print(f"Found {len(alices)} users named Alice: {alices}")
    
    # Find users by codename
    bob = user_locator.get_entity_by_codename(codename2)
    print(f"User with codename {codename2}: {bob}")
    
    # Example 2: Product locator using 'title' attribute
    product_locator = JustLocator[Product](entity_config_identifier_attr="title")
    
    product1 = Product("Laptop", 999.99)
    product2 = Product("Mouse", 25.50)
    
    prod_code1 = product_locator.publish_entity(product1)
    prod_code2 = product_locator.publish_entity(product2)
    
    print(f"\nRegistered products with codenames: {prod_code1}, {prod_code2}")
    
    # Find product by config identifier (title)
    laptops = product_locator.get_entities_by_config_identifier("Laptop")
    print(f"Found laptops: {laptops}")
    
    # Example 3: Service locator using default fallback (class name)
    # Since Service doesn't have a 'name' attribute, it will use class name
    service_locator = JustLocator[Service]()  # Uses default 'name' attr
    
    service1 = Service("web-hosting", "Web hosting service")
    service2 = Service("email", "Email service")
    
    svc_code1 = service_locator.publish_entity(service1)
    svc_code2 = service_locator.publish_entity(service2)
    
    print(f"\nRegistered services with codenames: {svc_code1}, {svc_code2}")
    
    # Since services don't have 'name' attribute, they'll be grouped by class name
    services_by_class = service_locator.get_entities_by_config_identifier("Service")
    print(f"Found services by class name: {services_by_class}")
    
    # Example 4: Get identifier information
    user_identifier = user_locator.get_identifier_by_instance(user1)
    if user_identifier:
        print(f"\nUser identifier - Class: {user_identifier.entity_class.__name__}, "
              f"Codename: {user_identifier.entity_codename}, "
              f"Config ID: {user_identifier.entity_config_identifier}")
    
    # Example 5: Arbitrary search
    expensive_products = product_locator.arbitrary_search(
        Product, 
        lambda p: p.price > 100
    )
    print(f"\nExpensive products (>$100): {expensive_products}")
    
    # Example 6: Get all entities by class
    all_user_codenames = user_locator.get_entity_codenames_by_class()
    print(f"\nAll user codenames: {all_user_codenames}")
    
    # Example 7: Unpublish entities
    removed_count = user_locator.unpublish_entities_by_config_identifier("alice")
    print(f"\nRemoved {removed_count} users named Alice")
    
    remaining_users = user_locator.get_entities_by_config_identifier("alice")
    print(f"Remaining users named Alice: {remaining_users}")


if __name__ == "__main__":
    example_usage() 