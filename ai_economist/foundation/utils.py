# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import json
import os
import sys
from hashlib import sha512

import lz4.frame
# from Crypto.PublicKey import RSA

from ai_economist.foundation.base.base_env import BaseEnvironment


def save_episode_log(game_object, filepath, compression_level=16):
    """Save an lz4 compressed version of the dense log stored
    in the provided game object"""
    assert isinstance(game_object, BaseEnvironment)
    compression_level = int(compression_level)
    if compression_level < 0:
        compression_level = 0
    elif compression_level > 16:
        compression_level = 16

    with lz4.frame.open(
        filepath, mode="wb", compression_level=compression_level
    ) as log_file:
        log_bytes = bytes(
            json.dumps(
                game_object.previous_episode_dense_log, ensure_ascii=False
            ).encode("utf-8")
        )
        log_file.write(log_bytes)


def load_episode_log(filepath):
    """Load the dense log saved at provided filepath"""
    with lz4.frame.open(filepath, mode="rb") as log_file:
        log_bytes = log_file.read()
    return json.loads(log_bytes)


# def verify_activation_code():
#     """
#     Validate the user's activation code.
#     If the activation code is valid, also save it in a text file for future reference.
#     If the activation code is invalid, simply exit the program
#     """
#     path_to_activation_code_dir = os.path.dirname(os.path.abspath(__file__))

#     def validate_activation_code(code, msg=b"covid19 code activation"):
#         filepath = os.path.abspath(
#             os.path.join(
#                 path_to_activation_code_dir,
#                 "scenarios/covid19/key_to_check_activation_code_against",
#             )
#         )
#         with open(filepath, "r") as fp:
#             key_pair = RSA.import_key(fp.read())

#         hashed_msg = int.from_bytes(sha512(msg).digest(), byteorder="big")
#         signature = pow(hashed_msg, key_pair.d, key_pair.n)
#         try:
#             exp_from_code = int(code, 16)
#             hashed_msg_from_signature = pow(signature, exp_from_code, key_pair.n)

#             return hashed_msg == hashed_msg_from_signature
#         except ValueError:
#             return False

#     activation_code_filename = "activation_code.txt"

#     filepath = os.path.join(path_to_activation_code_dir, activation_code_filename)
#     if activation_code_filename in os.listdir(path_to_activation_code_dir):
#         print("Using the activation code already present in '{}'".format(filepath))
#         with open(filepath, "r") as fp:
#             activation_code = fp.read()
#             fp.close()
#         if validate_activation_code(activation_code):
#             return  # already activated
#         print(
#             "The activation code saved in '{}' is incorrect! "
#             "Please correct the activation code and try again.".format(filepath)
#         )
#         sys.exit(0)
#     else:
#         print(
#             "In order to run this simulation, you will need an activation code.\n"
#             "Please fill out the form at "
#             "https://forms.gle/dJ2gKDBqLDko1g7m7 and we will send you an "
#             "activation code to the provided email address.\n"
#         )
#         num_attempts = 5
#         attempt_num = 0
#         while attempt_num < num_attempts:
#             activation_code = input(
#                 f"Whenever you are ready, "
#                 "please enter the activation code: "
#                 f"(attempt {attempt_num + 1} / {num_attempts})"
#             )
#             attempt_num += 1
#             if validate_activation_code(activation_code):
#                 print(
#                     "Saving the activation code in '{}' for future "
#                     "use.".format(filepath)
#                 )
#                 with open(
#                     os.path.join(path_to_activation_code_dir, activation_code_filename),
#                     "w",
#                 ) as fp:
#                     fp.write(activation_code)
#                     fp.close()
#                 return
#             print("Incorrect activation code. Please try again.")
#         print(
#             "You have had {} attempts to provide the activate code. Unfortunately, "
#             "none of the activation code(s) you provided could be validated. "
#             "Exiting...".format(num_attempts)
#         )
#         sys.exit(0)
