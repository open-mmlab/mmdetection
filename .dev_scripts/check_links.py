# Modified from:
# https://github.com/allenai/allennlp/blob/main/scripts/check_links.py

import argparse
import logging
import os
import pathlib
import re
import sys
from multiprocessing.dummy import Pool
from typing import NamedTuple, Optional, Tuple

import requests
from mmengine.logging import MMLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Goes through all the inline-links '
        'in markdown files and reports the breakages')
    parser.add_argument(
        '--num-threads',
        type=int,
        default=100,
        help='Number of processes to confirm the link')
    parser.add_argument('--https-proxy', type=str, help='https proxy')
    parser.add_argument(
        '--out',
        type=str,
        default='link_reports.txt',
        help='output path of reports')
    args = parser.parse_args()
    return args


OK_STATUS_CODES = (
    200,
    401,  # the resource exists but may require some sort of login.
    403,  # ^ same
    405,  # HEAD method not allowed.
    # the resource exists, but our default 'Accept-' header may not
    # match what the server can provide.
    406,
)


class MatchTuple(NamedTuple):
    source: str
    name: str
    link: str


def check_link(
        match_tuple: MatchTuple,
        http_session: requests.Session,
        logger: logging = None) -> Tuple[MatchTuple, bool, Optional[str]]:
    reason: Optional[str] = None
    if match_tuple.link.startswith('http'):
        result_ok, reason = check_url(match_tuple, http_session)
    else:
        result_ok = check_path(match_tuple)
    if logger is None:
        print(f"  {'✓' if result_ok else '✗'} {match_tuple.link}")
    else:
        logger.info(f"  {'✓' if result_ok else '✗'} {match_tuple.link}")
    return match_tuple, result_ok, reason


def check_url(match_tuple: MatchTuple,
              http_session: requests.Session) -> Tuple[bool, str]:
    """Check if a URL is reachable."""
    try:
        result = http_session.head(
            match_tuple.link, timeout=5, allow_redirects=True)
        return (
            result.ok or result.status_code in OK_STATUS_CODES,
            f'status code = {result.status_code}',
        )
    except (requests.ConnectionError, requests.Timeout):
        return False, 'connection error'


def check_path(match_tuple: MatchTuple) -> bool:
    """Check if a file in this repository exists."""
    relative_path = match_tuple.link.split('#')[0]
    full_path = os.path.join(
        os.path.dirname(str(match_tuple.source)), relative_path)
    return os.path.exists(full_path)


def main():
    args = parse_args()

    # setup logger
    logger = MMLogger.get_instance(name='mmdet', log_file=args.out)

    # setup https_proxy
    if args.https_proxy:
        os.environ['https_proxy'] = args.https_proxy

    # setup http_session
    http_session = requests.Session()
    for resource_prefix in ('http://', 'https://'):
        http_session.mount(
            resource_prefix,
            requests.adapters.HTTPAdapter(
                max_retries=5,
                pool_connections=20,
                pool_maxsize=args.num_threads),
        )

    logger.info('Finding all markdown files in the current directory...')

    project_root = (pathlib.Path(__file__).parent / '..').resolve()
    markdown_files = project_root.glob('**/*.md')

    all_matches = set()
    url_regex = re.compile(r'\[([^!][^\]]+)\]\(([^)(]+)\)')
    for markdown_file in markdown_files:
        with open(markdown_file) as handle:
            for line in handle.readlines():
                matches = url_regex.findall(line)
                for name, link in matches:
                    if 'localhost' not in link:
                        all_matches.add(
                            MatchTuple(
                                source=str(markdown_file),
                                name=name,
                                link=link))

    logger.info(f'  {len(all_matches)} markdown files found')
    logger.info('Checking to make sure we can retrieve each link...')

    with Pool(processes=args.num_threads) as pool:
        results = pool.starmap(check_link, [(match, http_session, logger)
                                            for match in list(all_matches)])

    # collect unreachable results
    unreachable_results = [(match_tuple, reason)
                           for match_tuple, success, reason in results
                           if not success]

    if unreachable_results:
        logger.info('================================================')
        logger.info(f'Unreachable links ({len(unreachable_results)}):')
        for match_tuple, reason in unreachable_results:
            logger.info('  > Source: ' + match_tuple.source)
            logger.info('    Name: ' + match_tuple.name)
            logger.info('    Link: ' + match_tuple.link)
            if reason is not None:
                logger.info('    Reason: ' + reason)
        sys.exit(1)
    logger.info('No Unreachable link found.')


if __name__ == '__main__':
    main()
