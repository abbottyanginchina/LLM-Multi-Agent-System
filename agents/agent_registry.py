from class_registry import ClassRegistry

class AgentRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def get(cls, *args, **kwargs):
        return cls.registry.get(*args, **kwargs)

