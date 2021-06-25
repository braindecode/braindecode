{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {#
      filter out inherited methods (such as the one from torch.nn.Module)
      and keep the only useful special methods.
   #}
   {% set to_display_methods = [] %}
   {% set include_special_methods = [
      '__contains__', '__getitem__', '__iter__', '__len__', '__add__',
      '__sub__', '__mul__', '__div__', '__neg__', '__hash__'
   ] %}
   {% for item in methods %}
      {% if item not in inherited_members and (
             not item.startswith('_') or item in include_special_methods) %}
         {% set __  = to_display_methods.append(item) %}
      {% endif %}
   {% endfor %}
   {% if to_display_methods %}
   {% block methods %}
   .. rubric:: Methods

   {% for item in to_display_methods %}
   .. automethod:: {{ item }}

   {% endfor %}
   {% endblock %}
   {% endif %}

.. include:: {{fullname}}.examples

.. raw:: html

    <div style='clear:both'></div>
