all_cell_features = ["is_blank", "bold_font", "below_blank", "has_merge_cell", "above_alpha", "left_align",
                     "right_blank", "above_blank", "above_num", "above_alphanum", "right_align", "underline_font",
                     "below_num", "left_alpha", "above_in_header", "left_num", "all_small", "is_alpha", "right_num",
                     "text_in_header", "is_num"]

default_header_values = ["%", "area", "author", "average", "avg", "capacity", "category", "city", "collection", "color",
                         "comment", "count", "country", "county", "date", "day", "depth", "description", "edition",
                         "family", "height", "hour", "-hr", " hr", "_hr", " id", "_id", "-id", "id_", "id-", "id ",
                         "index", "isbn", "key", "kind", "label", "master", "max", "maximum", "mean", "measure",
                         "median", "min", "minimum", " mode", "-mode", "_mode", "month", "name", "nitrate", "nitrite",
                         "percent", "qty", "quantity", "range", "rate", "record", "sample", "species", "standard",
                         "state", "station", "subject", "time", "title", "tot.", "total", "type", "variance", "volume",
                         "week", "eidth", "year", "yr."]

# null_default feature has been discarded
default_null_values = {"nil", "Nil", "NIL", "niL", "nIl", "NIl", "NiL", "null", "Null", "NuLL", "NUll",
                       "nULL", "NULL", "na", "NA", "Na", " ", "#", "VOID", "void", "Void", "(na)", "(Na)", "(NA)","n/a", "N/a", "N/A", "(n/a)", "(N/a)", "(N/A)", "(x)", "(X)", "-", " - ", " -", "- ", "--", "...", "No data", "No reading"}
