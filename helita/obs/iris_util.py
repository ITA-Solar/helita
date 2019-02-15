"""
Set of utility programs for IRIS.
"""
import os
import re
import io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from glob import glob

# pylint: disable=F0401,E0611,E1103
from urllib.request import urlopen
from urllib.parse import urljoin, urlparse
from urllib.error import HTTPError, URLError


def iris_timeline_parse(timeline_file):
    """
    Parses an IRIS timeline file (SCI format) into a structured array. This
    version outputs a strucured array instead of a pandas DataSet.

    Parameters
    ----------
    timeline_file - string
        Filename with timeline file, or URL to the file.

    Returns
    -------
    result - pandas.DataFrame
        DataFrame with timeline.
    """
    from sunpy.time import parse_time
    data = []
    slews = []
    curr_slew = np.array([np.nan, np.nan])
    line_pat = re.compile('.+OBSID=.+rpt.+endtime', re.IGNORECASE)
    slew_pat = re.compile('.+I_EVENT_MESSAGE.+MSG="SLEW*', re.IGNORECASE)
    if urlparse(timeline_file).netloc == '':   # local file
        file_obj = open(timeline_file, 'r')
    else:                                      # network location
        try:
            tmp = urlopen(timeline_file).read()
            file_obj = io.StringIO(tmp)
        except (HTTPError, URLError):
            raise EOFError(('iris_timeline_parse: could not open the '
                            'following file:\n' + timeline_file))
    for line in file_obj:
        if slew_pat.match(line):
            tmp = line.split('=')[1].replace('"', '').strip('SLEW_').split('_')
            curr_slew = np.array(tmp).astype('f')
        if line_pat.match(line):
            data.append(line.replace('//', '').replace(' x ', ', ').strip())
            slews.append(curr_slew)  # include most up to date slew
    file_obj.close()
    if len(data) == 0:
        raise EOFError(('iris_timeline_parse: could not find any'
                        ' observations in:\n' + str(timeline_file)))
    arr_type = [('date_obs', 'datetime64[us]'), ('date_end', 'datetime64[us]'),
                ('obsid', 'i8'), ('repeats', 'i4'), ('duration', 'f'),
                ('size', 'f'), ('description', '|S200'), ('xpos', 'f'),
                ('ypos', 'f'), ('timeline_name', '|S200')]
    result = np.zeros(len(data), dtype=arr_type)
    result['timeline_name'] = timeline_file
    for i, line in enumerate(data):
        date_tmp = line.split()[0]
        if date_tmp[-2:] == '60':   # deal with non-compliant second formats
            date_tmp = date_tmp[:-2] + '59.999999'
        result[i]['date_obs'] = parse_time(date_tmp)
        tmp = line.replace(' Mbits, end', ', end')  # Remove new Mbits size str
        tmp = tmp.split('desc=')
        result[i]['description'] = tmp[1]
        tmp = tmp[0]
        tmp = [k.split('=')[-1] for k in ' '.join(tmp.split()[1:]).split(',')]
        result[i]['obsid'] = int(tmp[0])
        result[i]['repeats'] = int(tmp[1])
        result[i]['duration'] = float(tmp[2][:-1])
        result[i]['size'] = float(tmp[3])
        tmp = tmp[4].split()
        result[i]['date_end'] = parse_time(date_tmp[:9] + tmp[-1]) + \
            timedelta(days=int(tmp[0].strip('+')))
        result[i]['xpos'] = slews[i][0]
        result[i]['ypos'] = slews[i][1]
    return pd.DataFrame(result)  # order by date_obs


def get_iris_timeline(date_start, date_end, path=None, fmt='%Y/%m/%d',
                      pattern='.*IRIS_science_timeline.+txt'):
    """
    Gets IRIS timelines for a given time period.
    """
    if path is None:
        path = ('http://iris.lmsal.com/health-safety/timeline/'
                'iris_tim_archive/')
    print('Locating files...')
    file_obj = FileCrawler(date_start, date_end, path, pattern, fmt)
    result = pd.DataFrame()
    for tfile in file_obj.files:
        try:
            print('Parsing:\n' + tfile)
            timeline = iris_timeline_parse(tfile)
            result = result.append(timeline)
        except EOFError:
            print('get_iris_timeline: could not read timeline data from:\n' +
                  tfile)
    return result


def get_iris_files(date_start, date_end, pattern='iris.*.fits', base='level1',
                   path='/Users/tiago/data/IRIS/data/'):
    """
    Gets list of IRIS observations for a given time period.

    Parameters
    ----------
    date_start : str or datetime object
        Starting date to search
    date_end : str or datetime object
        Ending date to search
    path : str
        Base path to look into
    pattern : str
        Regular expression used to match file names.

    Returns
    -------
    files : list
        List of strings with matching file names.
    """
    file_path = os.path.join(path, base)
    file_obj = FileCrawler(date_start, date_end, file_path, pattern,
                           fmt='%Y/%m/%d/H%H%M')
    return file_obj.files


class FileCrawler(object):
    """
    Crawls through file names in a local or remote (http) path.

    Parameters
    ----------
    date_start : str or datetime object
        Starting date to search
    date_end : str or datetime object
        Ending date to search
    path : str
        Base path to look into
    pattern : str
        Regular expression used to match file names.
    recursive: bool
        If True, will recursively search subdirectories of dates.

    Attributes
    ----------
    date_start : str or datetime object
        Starting date given as input
    date_end : str or datetime object
        Ending date given as input
    paths : list
        List of file paths given the supplied dates
    files : list
        List of file names given the supplied path, dates, and pattern

    Methods
    -------
    get_remote_paths(date_start, date_end, path, fmt='%Y%m%d')
        Finds existing remote paths within specified dates in path, given fmt.
    get_remote_files(path, pattern)
        Finds existing remote files within specified path matching pattern.
    """

    def __init__(self, date_start, date_end, path, pattern, fmt='%Y%m%d',
                 verbose=False):
        self.date_start = date_start
        self.date_end = date_end
        self.paths = self.get_paths(date_start, date_end, path, fmt)
        if verbose:
            print('Found the following paths:')
            for item in self.paths:
                print(item)
        self.files = []
        for item in self.paths:
            self.files += self.get_files(item, pattern)
        if verbose:
            print('Found the following files:')
            for item in self.files:
                print(item)

    @classmethod
    def get_paths(cls, date_start, date_end, path, fmt='%Y%m%d'):
        """
        Gets paths within specified date range.

        Parameters
        ----------
        date_start : str or datetime object
            Starting date to search
        date_end : str or datetime object
            Ending date to search
        path : str
            Base path where to look for locations (if starts with http,
            remote search will be done)
        format : str
            datetime format string for date in directories.

        Returns
        -------
        dates - list
            List with path locations (local directories or remote paths)
        """
        from sunpy.time import parse_time
        dates = []
        date_start = parse_time(date_start)
        date_end = parse_time(date_end)
        curr = date_start
        if '%H' in fmt:
            incr = [0, 1]   # increment only hours
        else:
            incr = [1, 0]   # increment only days
        if urlparse(path).netloc == '':   # local file
            while curr <= date_end:
                curr_path = os.path.join(path, datetime.strftime(curr, fmt))
                curr += timedelta(days=incr[0], hours=incr[1])
                if os.path.isdir(curr_path):
                    dates.append(curr_path)
        else:                             # network location
            while curr <= date_end:
                curr_path = urljoin(path, datetime.strftime(curr, fmt) + '/')
                curr += timedelta(days=incr[0], hours=incr[1])
                try:
                    urlopen(curr_path)
                    dates.append(curr_path)
                except (HTTPError, URLError):
                    continue
        return dates

    @classmethod
    def get_files(cls, path, pattern):
        """
        Obtains local or remote files patching a pattern.

        Parameters
        ----------
        path : str
            Local directory or remote URL (e.g. 'http://www.google.com/test/')
        pattern : str
            Regular expression to be matched in href link names.

        Returns
        -------
        files : list
            List of strings. Each string has the path for the files matching
            the pattern (and are made sure exist).

        .. todo:: add recursive option, add option for FTP
        """
        from bs4 import BeautifulSoup
        files = []
        pat_re = re.compile(pattern, re.IGNORECASE)
        if urlparse(path).scheme == '':   # local file
            all_files = glob(path + '/*')
            for item in all_files:
                if pat_re.match(item) and os.path.isfile(item):
                    files.append(item)
        elif urlparse(path).scheme == 'http':
            soup = BeautifulSoup(urlopen(path).read())
            for link in soup.find_all('a'):
                if pat_re.match(link.get('href')):
                    file_url = urljoin(path, link.get('href'))
                    try:   # Add only links that exist
                        urlopen(file_url)
                        files.append(file_url)
                    except (HTTPError, URLError):
                        pass
        elif urlparse(path).scheme == 'ftp':
            raise NotImplementedError('ftp not yet supported...')
        return files
