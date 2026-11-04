VERSION = 'v5.6.1'
# Task 12: the upstream patch identifiers are removed. They named release assets in the
# project's patch channel, which this fork neither checks nor installs; UVR.py:33 still
# imports these three names (current_patch feeds the license text at UVR.py:3911), so the
# names stay and the values point at nothing.
PATCH = 'no-update-channel'
PATCH_MAC = 'no-update-channel'
PATCH_LINUX = 'no-update-channel'
