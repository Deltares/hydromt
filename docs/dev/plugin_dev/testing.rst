.. _test_your_plugin:

Testing your plugin
===================

HydroMT Core offers some functionalities that can help you test your plugins, as well as
offering some examples on how to do it. Below are listed some tips and tricks to help
test your plugin and to help your users test their code using your plugin.


Testing model components
------------------------

When you implement a ``ModelComponent`` we very strongly encourage you to write a
``test_equal`` method on your component that should test for **deep** equality. This
means that it shouldn't just test whether it's literally the same python object, but
should actually attempt to test whether the attributes and any data held by it are the
same. This is not only important for your own testing, but also for your users, since
they might want to use that functionality to test their own code.

The function should have the following signature:

.. code-block:: python

    def test_equal(self, other: "ModelComponent") -> tuple[bool, Dict[str, str]]:
        ...

The return signature is a boolean indicating whether the equality is
true or not (as usual) and a dictionary. In this dictionary you can add any error
messages of whatever keys you'd like. If possible, it is good for usability if you can
report as many issues as possible in one go, so the user doesn't have to run the tests
over and over again.

As an example, suppose we have a ``RainfallComponent`` that must have a ``time`` dimension,
and ``x`` dimension, and additionally the data inside it must have the correct
``crs``. A function for that might look like this:


.. code-block:: python

    class RainfallComponent(ModelComponent):

        def test_equal(self, other: "ModelComponent") -> tuple[bool, Dict[str, str]]:
            errors: Dict[str, str] = {}
            if not isinstance(other, self.__class__):
                errors['type'] = """Other is not of type RainfallComponent and therfore
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

If all the components you use have a ``test_equal`` function defined, then testing for
model equality should be relatively simple. The core base model also has a
``test_equal`` function defined that will test all the components against each other so
if that is all you require you can simply use that function. If you wish to do
additional checks you can override this method and simply call
``super().test_equal(other)`` and do whatever checks you'd like after that.


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

To ensure your plugin is tested against multiple python versions, and both the latest
released version of HydroMT core and the latest development version, you can set up a
GitHub Actions workflow. This workflow will automatically run your test suite in different
environments whenever you push changes to your repository.

Here's a basic outline of how to set up a GitHub Actions workflow for your plugin:

1. In the pyproject.toml / pixi.toml of your plugin, ensure you have a test command defined that runs your tests.
Additionally, you should define the features and environments for the various Python versions and dependency versions you want to test.
A basic example:

.. literalinclude:: example_pixi.toml
  :language: toml

2. Create a new file in your repository at `.github/workflows/test.yml`, add the following content and update it as needed:

.. literalinclude:: example_test.yml
  :language: yaml

3. Commit and push the `test.yml` file to your repository.

This workflow will run your tests in different operating systems (Ubuntu and Windows), four different
Python versions (3.10, 3.11, 3.12, and 3.13), and against both the latest released version of HydroMT and the latest development version.
You can customize the workflow further based on your specific testing requirements.

Adding your plugin to the compatibility test bench
--------------------------------------------------

Before each HydroMT core release, we run an automated downstream compatibility test against a selection of mature plugins.
The goal is to detect breaking API changes and dependency conflicts before a release is published.

Two installation modes are executed.

In the first mode, HydroMT is installed with --no-deps.
This upgrades only the HydroMT wheel while keeping the plugin's environment exactly as previously solved.
This simulates a user upgrading HydroMT inside an already existing environment without re-solving dependencies.
This run answers the question: does the new HydroMT version remain compatible with an existing plugin environment?
Failures in this mode indicate that the upgrade is not drop-in compatible.
These require review and a conscious decision, but are not automatically considered release blockers.

In the second mode, HydroMT is installed allowing dependency updates.
This permits the resolver to adjust third-party packages if required by the new HydroMT version.
This simulates a clean installation or a full dependency resolution.
This run answers the question: can HydroMT and the plugin still coexist in a consistently solved environment?
Failures in this mode indicate a broken dependency graph or a fundamental incompatibility and are treated as release blockers.

For meaningful participation in the compatibility test bench, your plugin must:

1. Provide a Pixi environment that can be installed in a locked and reproducible way.
2. Define a test task that runs the full test suite via Pytest.
3. Does not require any interactive input or external services during testing.

If a compatibility failure is detected, the core and plugin maintainers should coordinate.
Depending on the nature of the issue, the resolution may involve restoring backward compatibility in HydroMT,
relaxing overly strict dependency constraints in the plugin, or preparing a plugin update aligned with the new HydroMT release.

The purpose of this test bench is not to guarantee that plugins never need updates.
Rather it is to ensure that any required changes are identified and handled deliberately rather than discovered by end users after release.
