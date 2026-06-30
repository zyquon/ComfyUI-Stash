# pyright: reportMissingImports=false
import os
import re
import json

from server import PromptServer

EXTENSION_DIR = os.path.dirname(os.path.abspath(__file__))
COMFYUI_DIR = os.path.dirname(os.path.dirname(EXTENSION_DIR))

DEFAULT_NAMESPACE = f'zyquon.ComfyUI-Stash'

# Unfortunately, these are shared with the JS code not DRY, because it seems
# that default values do not get written to the settings.json file, so Python needs to know them too.
DEFAULT = {
    'api_url': 'http://localhost:9999/graphql',
}

class Settings:
    def __init__(self, namespace=DEFAULT_NAMESPACE):
        self.settings_filepath = None
        self.namespace = namespace

    def get_settings_filepath(self):
        """Get the settings file path for the default user"""
        if self.settings_filepath:
            return self.settings_filepath

        user_manager = PromptServer.instance.user_manager
        users_dir = os.path.dirname(user_manager.get_users_file())
        # Ignore internal dirs like ComfyUI-Manager's `__manager`; prefer the
        # standard `default` user, since picking listdir()[0] is order-dependent.
        user_dirs = [
            d for d in os.listdir(users_dir)
            if os.path.isdir(os.path.join(users_dir, d)) and not d.startswith('_')
        ]
        if 'default' in user_dirs:
            user = 'default'
        elif user_dirs:
            user = user_dirs[0]
        else:
            print(f"Create user directory: {'default'!r}")
            user = 'default'
            os.mkdir(os.path.join(users_dir, user))

        self.settings_filepath = os.path.join(users_dir, user, "comfy.settings.json")
        return self.settings_filepath

    def get_settings(self):
        """Read the settings file"""
        settings = None
        filepath = self.get_settings_filepath()
        try:
            with open(filepath, "r") as file:
                config = json.load(file)
        except FileNotFoundError:
            print(f'WARNING Settings file not found: {filepath!r}')
            pass
        except json.JSONDecodeError as e:
            print(f'ERROR decoding JSON in settings file {filepath!r}: {e}')
            settings = None
        else:
            settings = DEFAULT.copy()
            for key, value in config.items():
                # Filter for only the settings relevant to this extension, and strip the namespace prefix.
                if key.startswith(f'{self.namespace}.'):
                    new_key = key[len(self.namespace) + 1:]
                    settings[new_key] = value
        return settings