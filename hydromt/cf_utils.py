import os, logging, zipfile
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Callable
import re
import shutil

logger = logging.getLogger("Utils")
logger.setLevel(logging.INFO)

class Utils():

    @staticmethod
    def get_logger(log_dir_path: str, log_name: str, log_level: logging.Logger = logging.INFO):
        """
        Creates log file
        :param log_dir_path: path where log file is written
        :param log_name: name of the logger as shown in log file
        :param log_level: default loglevel applied
        :return: logger class
        """
        logger = logging.getLogger(log_name)
        logger.setLevel(log_level)
        log_path = os.path.join(log_dir_path, "{}_log.txt".format(log_name))
        if os.path.exists(log_path):
            os.remove(os.path.join(log_dir_path, "{}_log.txt".format(log_name)))
        formatter = logging.Formatter('%(name)s - %(asctime)s %(levelname)s %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        ch = logging.FileHandler(os.path.join(log_dir_path, log_name + '.log'), mode='w')
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    @staticmethod
    def get_dir(base: str, subfolders: List[str]=[]) -> Path:
        """
        Static methods returns WindowsPath-object of directory as listed in configuration, if needed supplemented with sub-folders.
        Checks for folder existence and creates new if needed
        :param base: base folder from which sub-folders are checked/created
        :param subfolders: Fews-config subfolders
        :return: WindowsPath-object
        """
        if not Path(base).exists():
            Path(base).mkdir()
        if len(subfolders) == 0:
            return Path(base)
        else:
            folder = Path(base)
            for sub in subfolders:
                folder = Path(folder).joinpath(sub)
                if not folder.exists():
                    folder.mkdir()
            return folder

    @staticmethod
    def clean_dir(basedir: Path):
        """
        Cleans all files and sub-folders of directory
        :param basedir: path - folder to clear
        :return:
        """
        sub_dirs = [x for x in basedir.iterdir() if x.is_dir()]
        for sub_dir in sub_dirs:
            subsub_dirs = [x for x in sub_dir.iterdir() if x.is_dir()]
            if len(subsub_dirs)> 0:
                for subsub in subsub_dirs:
                    for file in subsub.iterdir():
                        file.unlink()
                    subsub.rmdir()
            for file in sub_dir.iterdir():
              file.unlink()
            sub_dir.rmdir()

    @staticmethod
    def create_zipfile(outfile: Path, files_to_append=List[str], add_dir_name:str=None, conditionalsuffix_sub_dir:Dict={}):
        """
        Creates zipfile-object by adding files listed and write do destination path
        :param outfile: path to output file
        :param files_to_append: list of files to include in zip file
        :param add_dir_name: add directory name when zipping
        :param conditionalsuffix_sub_dir: add sub-directory name (dict value) when file suffix in dict keys,
        only applied in combination with add_dir_name
        :return:
        """
        zipObj = zipfile.ZipFile(outfile, 'w', zipfile.ZIP_DEFLATED)
        for fileObj in files_to_append:
            if add_dir_name is None:
                zipObj.write(filename=fileObj, arcname=fileObj.name)
            if len(conditionalsuffix_sub_dir) > 0:
                for fsuffix in conditionalsuffix_sub_dir:
                    if fileObj.suffix == fsuffix:
                        zipObj.write(filename=fileObj, arcname='{}/{}/{}'
                                     .format(add_dir_name, conditionalsuffix_sub_dir[fsuffix], fileObj.name))
                    else:
                        zipObj.write(filename=fileObj, arcname='{}/{}'.format(add_dir_name, fileObj.name))
            else:
                zipObj.write(filename=fileObj, arcname='{}/{}'.format(add_dir_name,fileObj.name))
        zipObj.close()

    @staticmethod
    def write_ini_file(sections: Dict, ini_file: Path):

        """
        Reads parameter file and returns dataframe with:
        ID, NAME, SHORT_NAME, GROUP, TYPE, UNIT, VALUE_RESOLUTION
        :param ini:
        :return:
        """
        with open(ini_file, "w") as outf:
            for section in sections:
                outf.write('[' +  section + ']\n')
                for key in sections[section].keys():
                    outf.write(key + '=' + sections[section][key] + '\n')

    @staticmethod
    def add_wgs84_projection_file(fname: Path):
        """
        Add missing projection file, expressed in WGS 1984
        :param fname:
        :return:
        """
        proj_str = \
            'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
        proj_file = open(fname, "w")
        proj_file.write(proj_str)
        proj_file.close()

    @staticmethod
    def copy_basefiles(source_dir, destination_dir, dirs_exist_ok=True):
        """
        Copies base configuration to a specific goal directory
        :param cf_baseconfig_path: path with base configuration
        :param destination_path: path where final configuration is stored
        :return: copied base files
        """
        assert source_dir.exists(), f"Cannot find source directory for base configuration: {source_dir}"
        assert destination_dir.exists(), f"Cannot find destination directory for base configuration: {destination_dir}"
        shutil.copytree(Path(source_dir).resolve(), Path(destination_dir).resolve(),
                        dirs_exist_ok=dirs_exist_ok)

    @staticmethod
    def replace_tags_by_file(source_file, destination_file, tag_dict):
        """
        replaces tags in a source file templates and copies the filled source file to a destination file
        :param source_file: path to file that contains tags and therefore serves as template
        :param destination_file: path to file with tags from source file replaced by values
        :return: specific filled templates
        """

        if Path(destination_file).suffix == ".xml":
            destination_file = destination_file.parents[0]

        assert source_file.exists(), f"Cannot find source file: {source_file}"
        if not Path(destination_file).exists():
            Path(destination_file).mkdir(parents=True)

        destination_file_add = shutil.copy(source_file, destination_file, follow_symlinks=True)
        file_name = Path(destination_file_add).name

        with open(destination_file_add, 'r') as f:
            template = f.read()
            template_filled = template
            file_name_filled = file_name
            for key, value in tag_dict.items():
                file_name_filled = re.sub("{" + key + "}", value, file_name_filled)
                template_filled = re.sub("{" + key + "}", value, template_filled)
                with open(destination_file_add, 'w') as ff:
                    ff.write(template_filled)

        old_file_name = Path(destination_file / file_name)
        new_file_name = Path(destination_file / file_name_filled)

        if not Path(new_file_name).exists():
            old_file_name.rename(new_file_name)

    @staticmethod
    def replace_tags_by_filetree(source_dir_templ, destination_dir, tag_dict):
        """
        replaces tags in all source file templates in the file tree and save the filled templates in the respective destination pth
        :param source_file: source file that contains tags and therefore serves as template
        :param destination_file: file with tags from source file replaced by values
        :return: all filled templates
        """
        assert source_dir_templ.exists(), f"Cannot find source directory for templates: {source_dir_templ}"
        assert destination_dir.exists(), f"Cannot find destination directory: {destination_dir}"

        tag_files = list(source_dir_templ.glob("**/*{*}.xml"))
        for file_path in tag_files:
            tag_files_paths_filled = file_path
            for key, value in tag_dict.items():
                if key in str(tag_files_paths_filled):
                    tag_files_paths_filled = re.sub("{" + key + "}", value, str(tag_files_paths_filled))
                    if not "{" in tag_files_paths_filled:
                        config_dir = re.sub(f"^.*?\\\\Config", "Config", tag_files_paths_filled)
                        destination_dir_add = Path(destination_dir / config_dir)

                        Utils.replace_tags_by_file(file_path, destination_dir_add, tag_dict)