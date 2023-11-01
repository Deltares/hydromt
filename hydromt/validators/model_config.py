from typing import Any, Dict

from pydantic import BaseModel


class HydromtStep(BaseModel):
    # fn: Callable
    # args: Dict[str, Any]
    fn: str
    args: Dict[str, Any]

    @staticmethod
    def from_dict(input_dict):
        fn_name, arg_dict = next(iter(input_dict.items()))
        # TODO figure out where funcs are actually comming from
        # fn = getattr(hydromt, fn_name)
        # return HydromtStep(fn=getattr(hydromt, fn_name), args=arg_dict)
        return HydromtStep(fn=fn_name, args=arg_dict)
