def state_text_length(state_key: str):
    def length(state=None, **kwargs) -> float:
        if state is None:
            return 0.0
        return float(len(state.get(state_key) or ""))

    length.__name__ = f"length_{state_key}"
    return length
