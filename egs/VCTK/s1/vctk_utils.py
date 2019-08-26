#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import numpy as np
import os

db_speakers = ["p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232", "p233", "p234", "p236", "p237", "p238", "p239", "p240", "p241", "p243", "p244", "p245", "p246", "p247", "p248", "p249", "p250", "p251", "p252", "p253", "p254", "p255", "p256", "p257", "p258", "p259", "p260", "p261", "p262", "p263", "p264", "p265", "p266", "p267", "p268", "p269", "p270", "p271", "p272", "p273", "p274", "p275", "p276", "p277", "p278", "p279", "p280", "p281", "p282", "p283", "p284", "p285", "p286", "p287", "p288", "p292", "p293", "p294", "p295", "p297", "p298", "p299", "p300", "p301", "p302", "p303", "p304", "p305", "p306", "p307", "p308", "p310", "p311", "p312", "p313", "p314", "p315", "p316", "p317", "p318", "p323", "p326", "p329", "p330", "p333", "p334", "p335", "p336", "p339", "p340", "p341", "p343", "p345", "p347", "p351", "p360", "p361", "p362", "p363", "p364", "p374", "p376"]
db_speakers_dict = dict((k, i) for i, k in enumerate(db_speakers))
del db_speakers

db_speakers_English = ["p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232", "p233", "p236", "p239", "p240", "p243", "p244", "p250", "p254", "p256", "p257", "p258", "p259", "p267", "p268", "p269", "p270", "p273", "p274", "p276", "p277", "p278", "p279", "p282", "p286", "p287"]
db_speakers_English_dict = dict((k, i) for i, k in enumerate(db_speakers_English))


def id_name_to_speaker(id_name, length):
    try:
        return np.full((length, 1), db_speakers_dict[os.path.basename(id_name)[:4]], dtype=np.float32)
    except KeyError:
        raise KeyError("Unknown id_name: {}".format(id_name))


def id_name_to_speaker_English(id_name, length):
    try:
        return np.full((length, 1), db_speakers_English_dict[os.path.basename(id_name)[:4]], dtype=np.float32)
    except KeyError:
        raise KeyError("Unknown id_name: {}".format(id_name))


def main():
    length = 10
    assert((id_name_to_speaker("p226", length) == np.ones((length, 1), dtype=np.float32)).all())
    assert((id_name_to_speaker("p226/p226_001", length) == np.ones((length, 1), dtype=np.float32)).all())
    assert((id_name_to_speaker("p232", length) == np.full((length, 1), 7, dtype=np.float32)).all())
    assert((id_name_to_speaker("p252/p252_312", length) == np.full((length, 1), 25, dtype=np.float32)).all())
    try:
        id_name_to_speaker("unkown", length)  # This should throw an error.
    except KeyError:
        exit(0)

    exit(1)  # If no error is thrown, test failed.


if __name__ == "__main__":
    main()
