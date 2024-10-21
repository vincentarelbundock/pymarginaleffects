import narwhals as nw

def is_nw(obj: object) -> bool:
  return isinstance(obj, nw.dataframe.BaseFrame) or isinstance(obj, nw.Series) 