"""Typed KV routes for the tests' small in-memory pool doubles."""

from types import SimpleNamespace


class TypedPoolDouble:
    @property
    def layer_router(self):
        return self

    def read(self, layer_id):
        return SimpleNamespace(
            key=lambda: self.get_key_buffer(layer_id),
            value=lambda: self.get_value_buffer(layer_id),
        )

    def layer(self, layer_id):
        return SimpleNamespace(store=self.set_kv_buffer)
