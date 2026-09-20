"""newsscout.dashboard
~~~~~~~~~~~~~~~~~~~~~
Mobile-first web dashboard and REST API for NewsScout.
"""

from newsscout.dashboard.app import app, create_app

__all__ = ["app", "create_app"]