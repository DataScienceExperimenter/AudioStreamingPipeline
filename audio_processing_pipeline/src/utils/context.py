import asyncio
import contextvars
import uuid

# Context variables for tracking request context
request_id_var = contextvars.ContextVar('request_id', default=None)
user_id_var = contextvars.ContextVar('user_id', default=None)
session_var = contextvars.ContextVar('session', default=None)

class RequestContext:
    """Context manager for request tracking"""

    def __init__(self, user_id: str = None, session: str = None):
        self.request_id = str(uuid.uuid4())
        self.user_id = user_id
        self.session = session
        self.tokens = []

    async def __aenter__(self):
        # Set context variables
        self.tokens.append(request_id_var.set(self.request_id))
        self.tokens.append(user_id_var.set(self.user_id))
        self.tokens.append(session_var.set(self.session))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Reset context variables
        request_id_var.reset(self.tokens[0])
        user_id_var.reset(self.tokens[1])
        session_var.reset(self.tokens[2])
