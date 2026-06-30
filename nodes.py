import io
import os
import re
import numpy as np
import torch
import requests
import PIL.Image
import PIL.ImageOps
import PIL.ImageSequence

from server import PromptServer

from . import util
from .settings import Settings
from .stash_client import Stash, exceptions

NODE_CATEGORY = 'Stash'

class StashNode:
    """
    Connection to a Stash server. This is a required input for all other Stash nodes.
    """
    DESCRIPTION = __doc__
    CATEGORY = NODE_CATEGORY
    NAME = f'Stash Server'

    RETURN_NAMES = ("STASH", 'version')
    RETURN_TYPES = ("STASH", 'STRING' )

    @classmethod
    def INPUT_TYPES(cls):
        return { "required":{} }

    def __init__(self):
        self.stash = None
        self.api_key = None

    FUNCTION = 'run'
    def run(self):
        settings = Settings().get_settings()
        if not settings:
            raise ValueError(f'Could not read Comfy settings. Set the Stash API URL and Key in Comfy settings, then retry')

        api_url = settings.get('api_url')
        if not api_url:
            raise ValueError(f'You must set the Stash API URL in Comfy settings')

        self.api_key = settings.get('api_key')
        if not self.api_key:
            raise ValueError(f'You must set the Stash API Key in Comfy settings')

        self.stash = Stash(url=api_url, headers={"ApiKey": self.api_key})

        try:
            res = self.stash.version()
        except exceptions.GraphQLClientHttpError as e:
            print(f'\033[34m- \033[93m[WARNING]: Stash connection error: {e}\033[0m\n')
            raise Exception(f'Failed connection to Stash at {api_url}. Please check your settings and API key.')
        finally:
            pass

        version = res.version.version
        return (self.stash, version)

class StashImage:
    """
    Get an image from Stash by ID, search string, URL in the UI. Results are ORed; use the offset to choose the image
    """
    DESCRIPTION = __doc__
    CATEGORY = NODE_CATEGORY
    NAME = f'Stash Image'

    RETURN_NAMES = ("IMAGE", 'id' , 'count')
    RETURN_TYPES = ("IMAGE", 'INT', 'INT'  )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'stash': ('STASH', {'tooltip': 'A Stash server connection'}),
            },
            'optional': {
                'id_or_url': ('STRING', {
                    'default': '',
                    'tooltip': 'The image ID or a URL from the UI',
                }),
                'search': ('STRING', {
                    'default': '',
                    'tooltip': 'A search as when running in the Stash UI',
                }),
                'tags': ('STRING', {
                    'default': '',
                    'tooltip': 'One or more tags associated with the image, comma-separated',
                }),
                # TODO: Maybe just say the name of your Stash saved search and it can pull that and DTRT.
                'offset': ('INT', {
                    'default': 0,
                    'tooltip': 'Offset to use if the query returns more than one image (0 is the first image)',
                }),
            }
        }

    FUNCTION = 'run'
    def run(self, stash:Stash, offset, id_or_url, search, tags):
        # TODO I think this is incorrect. Returning None did not work but returning this empty tensor just
        # triggers an exception like no such index 0 from the IMAGE consumer node.
        empty = torch.empty((0, 0, 0, 3), dtype=torch.float32)

        tags = self.split_commas(tags)

        id_ur_url = id_or_url.strip()
        match = re.search(r'^https?://.+/images/(\d+)', id_or_url)
        if match:
            ids = [match.group(1)]
        else:
            ids = self.split_commas(id_or_url, isdigit=True)

        all_images = []
        if ids:
            print(f'Stash: Images by ID: {ids!r}')
            res = stash.images_by_ids(ids=ids)
            all_images += res.find_images.images
        if search:
            print(f'Stash: Images by search: {search!r}')
            res = stash.images_by_search(q=search)
            all_images += res.find_images.images
        if tags:
            print(f'Stash: Images by tags: {tags!r}')
            tag_ids = self.get_tag_ids(stash, *tags)
            res = stash.images_by_tag_ids(ids=tag_ids)
            all_images += res.find_images.images

        # De-dupe
        stash_images = []
        seen = set()
        for image in all_images:
            if image.id not in seen:
                seen.add(image.id)
                stash_images.append(image)

        match_count = len(stash_images)
        print(f'Stash: found images: {match_count}')
        if match_count == 0:
            return (empty, 0)

        try:
            stash_image = stash_images[offset]
        except IndexError:
            print(f'Stash: WARNING Offset {offset} is out of bounds for {len(stash_images)} images')
            return (empty, 0)

        print(f'Stash: found image: {stash_image.id}')

        # Pull the image data.
        # TODO: A cheeky optimization would be to check if the image exist locally, maybe only if the API URL indicates a local host.
        url = stash_image.paths.image
        headers = stash.headers.copy()
        with requests.get(url, headers=headers, timeout=30) as response:
            if response.status_code != 200:
                raise Exception(f'Got {response.status_code} in image {stash_image.id}: {url}')
            body = response.content

        image_set = PIL.Image.open(io.BytesIO(body)) # e.g. an animated webp

        w, h = None, None
        comfy_images = []
        for image in PIL.ImageSequence.Iterator(image_set):
            image = PIL.ImageOps.exif_transpose(image)
            if image.mode == 'I':
                image = image.point(lambda i: i * (1/255))
            image = image.convert('RGB')

            if w is None:
                w = image.size[0]
                h = image.size[1]
            if image.size[0] != w or image.size[1] != h:
                print(f'Warning: Image {image.id} has different dimensions ({image.size[0]}x{image.size[1]}) than the first frame ({w}x{h}), skipping.')
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            comfy_images.append(image)

        if len(comfy_images) > 1:
            output = torch.cat(comfy_images, dim=0)
        else:
            output = comfy_images[0]

        return (output, stash_image.id, match_count)

    def get_tag_ids(self, stash, *tag_names):
        tags_re = [ re.escape(tag_name.strip()) for tag_name in tag_names ]
        tags_re = r'^' + r'|'.join(tags_re) + r'$'
        res = stash.tags_by_regex(regex=tags_re)
        tags = res.find_tags.tags
        tag_ids = [ int(tag.id) for tag in tags ]
        return tag_ids

    def split_commas(self, vals, empty=False, isdigit=False):
        """Return a list of IDs or values from a comma-separated string."""
        if isinstance(vals, str):
            vals = vals.split(',')
        if not isinstance(vals, list):
            raise ValueError(f'Expected a string or list of IDs, got {type(vals)}')

        vals = [ v.strip() for v in vals ]
        if not empty:
            vals = [ v for v in vals if v ]
        if isdigit:
            vals = [ v for v in vals if v.isdigit() ]
        return vals

    # TODO, IS_CHANGED, maybe query for an etag or some checksum or timestamp of the image.
    # Remember this must return a value which will be compared to the previous return value, not a boolean.
    # @classmethod
    # def IS_CHANGED(s, image):
    #     image_path = folder_paths.get_annotated_filepath(image)
    #     m = hashlib.sha256()