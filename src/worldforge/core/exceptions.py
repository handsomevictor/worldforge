"""worldforge exception hierarchy."""


class WorldForgeError(Exception):
    """Base exception for all worldforge errors."""


class ConfigurationError(WorldForgeError):
    """Raised when simulation or component configuration is invalid."""


class SimulationError(WorldForgeError):
    """Raised when a fatal error occurs during simulation execution."""


class AgentError(WorldForgeError):
    """Raised when an agent operation fails."""


class EventOrderError(WorldForgeError):
    """Raised when an event would be scheduled in the past."""


class DistributionError(WorldForgeError):
    """Raised when distribution parameters are invalid or sampling fails."""
