#! /usr/bin/env python
import delayapi
from delayapi import app

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)