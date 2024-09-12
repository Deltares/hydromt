.. _best_practices:

Best Practices
===================


ModelComponents will recieve a weak reference back to the model they are a part of.
This means that though that reference, model components should be able to acccess
anything else you might need including, the data catalog, model information, and other
components. It does however mean that components cannot meaningfully exist outside of a
Model.
