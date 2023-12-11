# FinGPT: FinRL and ChatGPT integration prototype

This tool is a standalone incorporation of the FinRL source framework and ChatGPT api calls to create an explainable use-case for AI-managed financial portfolios.

# Install Directions for Windows only. Be weary this installs many packages.

Install Python 3.10 @ https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
	- Run Installer
	- Open 'Edit the system environment variables'
	- click 'environment variables'
	- Under system variables, edit PATH
	- Add new: C:\Users\'$User'\AppData\Local\Programs\Python\Python310 and C:\Users\'$User'\AppData\Local\Programs\Python\Python310\Scripts
Install Git @ https://git-scm.com/download/win
	- Run Installer
	- Open 'Edit the system environment variables'
	- click 'environment variables'
	- Under system variables, edit PATH
	- Add new: 'C:\Program Files\Git\cmd'
Unzip FinGPT.zip
Double-click 'installPackages.py'
	- If there are errors with this installation for box2d, you can extract the three zips in the box2d folder to:
		C:\Users\$(User)\AppData\Local\Programs\Python\Python310\Lib\site-packages

# Run application

1. Double-click FinGPT.pyw
2. Enter parameters:
	a. dates - START DATE MUST BE A MONDAY
	b. tickers
	c. cash available
	d. max stock
	e. risk factor - FinRL uses this factor  as "Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated"
	f. indicators selection
3. pull data button
4. train button - wait for console output before continuing
5. predict button
6. explain button
7. explain indicators button
etc. explain buttons can continue to be used

#Known Issues:

- Start date must be a Monday!
- Trade dates must be a range and not a single date.
- close_30_sma Does not work.
- close_60_sma Does not work.
- 'Explain' Function can only be used after a predit call.
- 'Explain Indicators' will output nonsense if not used after 'Explain'
- The ChatGPT API Key is will only be valid temporarily. A user must create their own key.

# Help

Author: Ryan Cathey
Email: RyanCCat@gmail.com