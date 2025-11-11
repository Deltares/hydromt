.. _test_your_plugin:

Testing your plugin
===================

HydroMT Core offers some functionalities that can help you test your plugins, as well as
offering some examples on how to do it. Below are listed some tips and tricks to help
test your plugin and to help your users test their code using your plugin.


Testing model components
------------------------

When you implement a ``ModelComponent`` we very strongly encourage you to write a
``test_equals`` method on your component that should test for **deep** equality. This
means that it shouldn't just test whether it's literally the same python object, but
should actually attempt to test whether the attributes and any data held by it are the
same. This is not only important for your own testing, but also for your users, since
they might want to use that functionality to test their own code.

The function should have the following signature:

.. code-block:: python

    def test_equal(self, other: "ModelComponent") -> tuple[bool, Dict[str, str]]:
        ...

So counter to what we told you about processes this function **should** take the actual
model as input. The return signature is a boolean indicating whether the equality is
true or not (as usual) and a dictionary. In this dictionary you can add any error
messages of whatever keys you'd like. If possible, it is good for usability if you can
report as many issues as possible in one go, so the user doesn't have to run the tests
over and over again.

As an example, suppose we have a `RainFallComponent` that must have a `time` dimension,
and ``x`` dimension, and additionally the data inside it must have the correct
crs. A function for that might look like this:


.. code-block:: python

    class RainFallComponent(ModelComponent):

        def test_equal(self, other: "ModelComponent") -> tuple[bool, Dict[str, str]]:
            errors: Dict[str, str] = {}
            if not isinstance(other, self.__class__):
                errors['type'] = """Other is not of type RainFallComponent and therfore
                cannot be equal"""
                return (False, errors)

            if not 'time' in other.dims().keys():
                errors['no_time_dim'] = """Component does not have the required time
                dimension"""
            else:
                if not self.data.sel(time=slice(None)).equals(other.data.sel(time=slice(None))):
                    errors['time_dim_not_equal'] = "time dimension data is not equal"

            if not 'x' in other.dims().keys():
                errors['no_x_dim'] = """Component does not have the required x
                dimension"""
            else:
                if not self.data.sel(x=slice(None)).equals(other.data.sel(x=slice(None))):
                    errors['x_dim_not_equal'] = "x dimension data is not equal"

            return len(errors) == 0, errors

Note that in the case the classes are not equal we return early since it probably
doesn't make sense to test for data equality on random classes. However, in the other
cases we check both the time and x dimension at the same time. This gives users as much
information about what is wrong as possible.

Testing models
--------------

If all the components you use have a ``test_equal`` function defined than testing for
model equality should be relatively simple. The core base model also has a
``test_equal`` function defined that will test all the components against each other so
if that is all you require you can simply use that function. If you wish to do
additional checks you can override this method and simply call
``super().test_equals(other)`` and do whatever checks you'd like after that.


Testing your plugin as a whole
------------------------------

Depending on how up to date with core developments you'd like to be it might be good to
test against both the latest released version of hydromt core (which is presumably what
your users will be using) as well as the latest version of Hydromt on the main branch
(the development version as it were). This can help you anticipate if core might release
features in the future that are incompatible for you and fix problems before they arise
for your users. This also gives you the opportunity to file bug reports with core to fix
things before they are released, which we highly encourage! If you use an
environment/package manager that supports this such as pixi then you can do this by
making a separate optional dependency-group for it, and simply run the test suite against
the different environments in your CI.


Setting up a GitHub actions workflow
-------------------------------------

To ensure your plugin is tested against multiple python versions, and both the latest released version of HydroMT core and the latest development version, you can set up a GitHub Actions workflow.
This workflow will automatically run your test suite in different environments whenever you push changes to your repository.

Here's a basic outline of how to set up a GitHub Actions workflow for your plugin:

1. Create a new file in your repository at `.github/workflows/test.yml`.

```yaml
name: Test Plugin

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
        hydromt-version: [latest, main]

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pixi install -e {{ matrix.hydromt-version }}


      - name: Run tests
        run: |
          pixi run test
```

2. Commit and push the `test.yml` file to your repository.

This workflow will run your tests in three different Python environments (3.8, 3.9, and 3.10) and against both the latest released version of HydroMT and the latest development version. You can customize the workflow further based on your specific testing requirements.

2. Add the following content to the `test.yml` file:

```yaml
name: Test Plugin

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
        hydromt-version: [latest, main]

    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install hydromt==${{ matrix.hydromt-version }}

      - name: Run tests
        run: |
          pytest tests/
```

3. Commit and push the `test.yml` file to your repository.

This workflow will run your tests in three different Python environments (3.8, 3.9, and 3.10) and against both the latest released version of HydroMT and the latest development version. You can customize the workflow further based on your specific testing requirements.
