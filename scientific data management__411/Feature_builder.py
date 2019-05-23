import re
import numpy as np
import cell
import datetime


class Feature_builder:
    nullDefault = []
    headers = []

    def init_headers(self):
        self.headers.append("%")
        self.headers.append("area")
        self.headers.append("author")
        self.headers.append("average")
        self.headers.append("avg")
        self.headers.append("capacity")
        self.headers.append("category")
        self.headers.append("city")
        self.headers.append("collection")
        self.headers.append("color")
        self.headers.append("comment")
        self.headers.append("count")
        self.headers.append("country")
        self.headers.append("county")
        self.headers.append("date")
        self.headers.append("day")
        self.headers.append("depth")
        self.headers.append("description")
        self.headers.append("edition")
        self.headers.append("family")
        self.headers.append("height")
        self.headers.append("hour")
        self.headers.append("-hr")
        self.headers.append(" hr")
        self.headers.append("_hr")
        self.headers.append(" id")
        self.headers.append("_id")
        self.headers.append("-id")
        self.headers.append("id_")
        self.headers.append("id-")
        self.headers.append("id ")
        self.headers.append("index")
        self.headers.append("isbn")
        self.headers.append("key")
        self.headers.append("kind")
        self.headers.append("label")
        self.headers.append("master")
        self.headers.append("max")
        self.headers.append("maximum")
        self.headers.append("mean")
        self.headers.append("measure")
        self.headers.append("median")
        self.headers.append("min")
        self.headers.append("minimum")
        self.headers.append(" mode")
        self.headers.append("-mode")
        self.headers.append("_mode")
        self.headers.append("month")
        self.headers.append("name")
        self.headers.append("nitrate")
        self.headers.append("nitrite")
        self.headers.append("percent")
        self.headers.append("qty")
        self.headers.append("quantity")
        self.headers.append("range")
        self.headers.append("rate")
        self.headers.append("record")
        self.headers.append("sample")
        self.headers.append("species")
        self.headers.append("standard")
        self.headers.append("state")
        self.headers.append("station")
        self.headers.append("subject")
        self.headers.append("time")
        self.headers.append("title")
        self.headers.append("tot.")
        self.headers.append("total")
        self.headers.append("type")
        self.headers.append("variance")
        self.headers.append("volume")
        self.headers.append("week")
        self.headers.append("eidth")
        self.headers.append("year")
        self.headers.append("yr.")

    def init_nullDefault(self):
        self.nullDefault.append("nil")
        self.nullDefault.append("Nil")
        self.nullDefault.append("NIL")
        self.nullDefault.append("0")
        self.nullDefault.append("00")
        self.nullDefault.append("niL")
        self.nullDefault.append("nIl")
        self.nullDefault.append("NIl")
        self.nullDefault.append("NiL")
        self.nullDefault.append("null")
        self.nullDefault.append("Null")
        self.nullDefault.append("NuLL")
        self.nullDefault.append("NUll")
        self.nullDefault.append("nULL")
        self.nullDefault.append("NULL")
        self.nullDefault.append("na")
        self.nullDefault.append("NA")
        self.nullDefault.append("Na")
        self.nullDefault.append(" ")
        self.nullDefault.append("#")
        self.nullDefault.append("VOID")
        self.nullDefault.append("void")
        self.nullDefault.append("Void")
        self.nullDefault.append("0%")
        self.nullDefault.append("(na)")
        self.nullDefault.append("(Na)")
        self.nullDefault.append("(NA)")
        self.nullDefault.append("n/a")
        self.nullDefault.append("N/a")
        self.nullDefault.append("N/A")
        self.nullDefault.append("(n/a)")
        self.nullDefault.append("(N/a)")
        self.nullDefault.append("(N/A)")
        self.nullDefault.append("(x)")
        self.nullDefault.append("(X)")
        self.nullDefault.append("-")
        self.nullDefault.append(" - ")
        self.nullDefault.append(" -")
        self.nullDefault.append("- ")
        self.nullDefault.append("--")
        self.nullDefault.append("z")
        self.nullDefault.append("Z")
        self.nullDefault.append("...")
        self.nullDefault.append("zero")
        self.nullDefault.append("Zero")
        self.nullDefault.append("ZERO")
        self.nullDefault.append("empty")
        self.nullDefault.append("No data")
        self.nullDefault.append("No reading")

    def Is_alphabet_Feature(self, text):
        if (text.isalpha()):
            return True
        return False

    def Is_alphaNumeric_Feature(self, text):
        if (text.isalnum()):
            return True
        return False

    def Is_textInHeader_Feature(self, text):
        temp_headers = []
        # if self.getCellTag() == "CH":
        #     temp_headers.append(self.get_val())
        self.init_headers()
        if (isinstance(text, str)):
            lowerText = text.lower()
        # for header in temp_headers:
        #     if text == header:
        #         return True
        #     print(lowerText)
            for header in Feature_builder.headers:

                if header in lowerText:
                    # print(header)
                    return True
            # print("yeah, I get here")
            if lowerText == "id":
                return True
            if lowerText == "mode":
                return True
            if lowerText == "hr":
                return True

        return False

    def Is_nullDefault_Feature(self, text):
        self.init_nullDefault()
        for item in Feature_builder.nullDefault:
            if text == item:
                return True
        return False

    def Is_allSmall_Feature(self, text):
        smallCnt = 0
        letterCnt = 0
        if isinstance(text, datetime.date):
            return True
        letterCnt = text.count(''.join(char for char in text if char.isalpha()))
        for i in range(len(text)):
            if text[i].isalpha and text[i].islower():
                smallCnt += 1
        if smallCnt == letterCnt and letterCnt > 0:
            return True
        return False

    def Is_allCapital_Feature(self, text):
        upperCnt = 0
        letterCnt = 0
        if isinstance(text, datetime.date):
            return False
        letterCnt = text.count(''.join(char for char in text if char.isalpha()))
        for i in range(len(text)):
            if text[i].isalpha and text[i].islower():
                upperCnt += 1
        if letterCnt == upperCnt and letterCnt > 0:
            return True
        return False

    def Is_startCapital_Feature(self, text):
        try:
            upper = text[0].isupper()
            if (upper):
                return True
            return False
        except:
            print("First character is not an alphabet")
            return False

    def contains_Colon_Feature(self, text):

        if bool(re.search("\:", str(text))):
            return True
        return False

    def contains_Special_Feature(self, text):
        if (bool(re.search("^[ \ta-zA-Z0-9(){}\[\]_=`~+|<>?.,/:;'\\]*[!@#$%^&*](.*)*$", str(text)))):
            return True
        return False

    def Is_textLength_Feature(self, text):
        if isinstance(text, datetime.date):
            return False
        if len(text) >= 60:
            return True
        return False

    def Is_inYearRange_Feature(self, text):
        try:
            year = np.int32(text)
            if year >= 1800 and year <= 2200:
                return True
            return False
        except ValueError:
            print("Input string is not a sequence of digits.")
            return False
        except OverflowError:
            print("The number cannot fit in an Int32.")
        except TypeError:
            print("The text is a datetime type")
            return False

