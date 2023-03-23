import time
import sys
import logging
import requests
import yaml

from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

with open("api_keys.yaml", "r") as yamlfile:
    api_keys = yaml.load(yamlfile, Loader=yaml.FullLoader)["keys"]


def is_number(val):
    if isinstance(val, bool):
        return False
    else:
        try:
            float(val)
            return True
        except ValueError:
            return False
        except TypeError:
            return False

class InvalidResponse(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InvalidRequest(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ApiRateError(Exception):
    def __init__(self, message, pass_on=None):
        self.message = message
        super().__init__(self.message, pass_on)


class DataUnavailableError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class FinancialModelingPrep:
    def __init__(self, number_workers=5):
        self.api_key = api_keys[0]
        self.base_path = "https://financialmodelingprep.com/api"
        self.fx_spot_rates = {}
        self.number_workers = number_workers

    def make_request(self, url, pass_to_multithreader=None):
        # pass_to_multithreader must be a list of the args that were passed to the function where this is called
        if pass_to_multithreader is None:
            logger.warning(
                f"call to make_request in class FinancialModelingPrep did not provide pass_to_multithreader, this may cause problems while multithreading. url: {url}")
        if pass_to_multithreader == "N/A":  # i can pass this string to avoid the warning but i dont want this to mess with the multithreader
            pass_to_multithreader = None
        try:
            response = requests.request("GET", url, timeout=2)
        except requests.exceptions.ConnectionError as e:
            time.sleep(1)
            raise ApiRateError(f"Connection error. url: {url}", pass_to_multithreader)
        except requests.exceptions.Timeout:
            raise ApiRateError(
                f"Connection timed out, this is likely connected to making too many requests. url: {url}",
                pass_to_multithreader)
        if response.status_code == 429:
            raise ApiRateError(f"API limit reached. url: {url}", pass_to_multithreader)
        else:
            response = response.json()

        if not response:
            raise InvalidResponse(f"Invalid Response from API for url <{url}>")
        elif "Error Message" in response:
            if "Too Many Requests" in response["Error Message"]:
                raise ApiRateError(f"API limit per millisecond exceeded. url: {url}", pass_to_multithreader)
            else:
                raise InvalidResponse(f'{response["Error Message"]}. url: {url}')
        elif "error" in response:
            if response["error"] == "Not found! Please check the date and symbol.":
                raise DataUnavailableError("No data available for this date and ticker symbol")
            elif "Error Message" in response:
                raise InvalidResponse(f'{response["Error Message"]}. url: {url}')
            else:
                raise InvalidResponse(f"Invalid response: {response}. url: {url}")
        else:
            return response

    def call_report_currency(self, ticker_symbol):
        url = f"{self.base_path}/v3/profile/{ticker_symbol}?apikey={self.api_key}"
        return self.make_request(url, [ticker_symbol])[0]["currency"], {"ticker_symbol": ticker_symbol}

    def convert_currency(self, convert_from, convert_to, values):
        # values can be a single number or an iterator of numbers
        if convert_from == convert_to:
            return values
        if convert_from + convert_to in self.fx_spot_rates:
            rate = self.fx_spot_rates[convert_from + convert_to]
        elif convert_to + convert_from in self.fx_spot_rates:
            rate = 1 / self.fx_spot_rates[convert_to + convert_from]
            self.fx_spot_rates[convert_from + convert_to] = rate
        else:
            url = f"{self.base_path}/v3/historical-chart/1min/{convert_from}{convert_to}?apikey={self.api_key}"
            rate = self.make_request(url, [convert_from, convert_to, values])[0]["close"]
            self.fx_spot_rates[convert_from + convert_to] = rate

        if is_number(values):
            return rate * values
        else:
            return [val * rate for val in values]

    def call_price(self, ticker_symbol, currency="USD"):
        url = self.base_path + f"/v3/quote-short/{ticker_symbol}?apikey={self.api_key}"
        price = self.make_request(url, [ticker_symbol, currency])[0]["price"]

        source_currency, meta = self.call_report_currency(ticker_symbol)

        return self.convert_currency(source_currency, currency, price), {"ticker_symbol": ticker_symbol,
                                                                        "currency": currency,
                                                                        "status": "ok"}

    def check_exists(self, ticker_symbol):
        try:
            self.call_profile(ticker_symbol)
            return True, {"status": "ok"}
        except (DataUnavailableError, InvalidResponse):
            return False, {"status": "nok"}

    def call_profile(self, ticker_symbol):
        url = self.base_path + f'/v3/profile/{ticker_symbol}?limit=100&apikey={self.api_key}'
        response = self.make_request(url, [ticker_symbol])
        return response[0], {"ticker_symbol": ticker_symbol, "status": "ok"}

    def call_close_timeseries(self, ticker_symbol):
        url = f"{self.base_path}/v3/historical-price-full/{ticker_symbol}?apikey={self.api_key}"
        response = self.make_request(url, [ticker_symbol])
        return response["historical"], {"status": "ok", "ticker_symbol": ticker_symbol}

    def call_all_available_tickers(self):
        url = self.base_path + "/v3/financial-statement-symbol-lists?apikey=" + self.api_key
        response = self.make_request(url, pass_to_multithreader="N/A")
        response.remove("Cash")
        return response, {"status": "ok"}


def multithreaded_request(function, *args, number_workers=None, verbose=True):
    """
    every *args argument must be a list of arguments to pass to the function, the i´th elements of all arg lists will
    be passed. The first argument after the function (second argument to this function) must be a list of
    ticker symbols
    """
    n_args = len(args)
    if number_workers is None:
        number_workers = function.__self__.number_workers
    (response, retry), meta = _execute_requests_multithreaded(function, args, number_workers, verbose=verbose)
    errors_last_try = None
    for i in range(10):
        # retry i times with lower number of workers
        number_workers = max(1, round(number_workers * 0.5))
        if not meta["n_api_rate_error"]:
            return response, {"status": "ok", "n_api_rate_errors": 0}

        print(f"  encountered {meta['n_api_rate_error']} api rate errors, retrying with {number_workers} workers")
        retry_args = [[] for _ in range(n_args)]
        for args in retry:
            for j, arg in enumerate(args):
                if not j + 1 <= n_args:
                    raise RuntimeError(
                        f"error while collecting failed calls with retry_args: {retry_args}. arg that failed: {arg}")
                retry_args[j].append(arg)
        (smaller_response, retry), meta = _execute_requests_multithreaded(function, retry_args, number_workers, verbose)
        response.update(smaller_response)
        if meta["n_api_rate_error"] == errors_last_try:
            print("aborting due to lack of improvement")
            break

    if meta["n_api_rate_error"] == 0:
        return response, {"status": "ok", "n_api_rate_error": 0}
    else:
        logger.warning(f"warning, there are {meta['n_api_rate_error']} api rate errors")
        return response, {"status": "warn", "api_rate_errors": meta['n_api_rate_error']}


def _execute_requests_multithreaded(function, args, number_workers, verbose):
    #print(function, args)
    executor = ThreadPoolExecutor(max_workers=number_workers)
    threads = []
    response = {}
    retry = []
    n_invalid_response = 0
    n_api_rate_error = 0
    n_calls = len(args[0])
    n_args = len(args)
    for arg_list in args:
        assert len(arg_list) == n_calls
    if n_args == 1:
        for i in range(n_calls):
            threads.append(executor.submit(function, args[0][i]))
    elif n_args == 2:
        for i in range(n_calls):
            threads.append(executor.submit(function, args[0][i], args[1][i]))
    elif n_args == 3:
        for i in range(n_calls):
            threads.append(executor.submit(function, args[0][i], args[1][i], args[2][i]))
    elif n_args == 4:
        for i in range(n_calls):
            threads.append(executor.submit(function, args[0][i], args[1][i], args[2][i], args[3][i]))
    elif n_args == 5:
        for i in range(n_calls):
            threads.append(executor.submit(function, args[0][i], args[1][i], args[2][i], args[3][i], args[4][i]))
    else:
        raise RuntimeError("I did not implement having this many args, just add a case :)")
    last_error = "None"
    for task in as_completed(threads):
        try:
            result, meta = task.result()
            assert isinstance(meta, dict)
            assert "status" in meta
            assert meta["status"] == "ok"
            response[meta["ticker_symbol"]] = result
        except (InvalidResponse, DataUnavailableError) as e:
            last_error = str(e).split("url")[0].replace("('", "")
            n_invalid_response += 1
        except ApiRateError as e:
            last_error = str(e).split("url")[0].replace("('", "")
            n_api_rate_error += 1
            arguments_for_retry = e.args[1]
            retry.append(arguments_for_retry)
        if verbose:
            sys.stdout.write(f"\r  n_successes: {len(response)}, n_failures: {n_invalid_response}, n_api_errors: {n_api_rate_error} out of {n_calls} last error: {last_error}")

    if verbose:
        print()
        sys.stdout.flush()
    return (response, retry), {"status": "ok",
                               "n_invalid_response": n_invalid_response,
                               "n_api_rate_error": n_api_rate_error,
                               "success_rate": len(response) / n_calls}

