
/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/


/**
 * AUTO-GENERATED FILE. DO NOT MODIFY.
 */

/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/
import { toCamelCase } from '../../util/format.js';
import env from 'zrender/lib/core/env.js';
/* global document */
export function shouldTooltipConfine(tooltipModel) {
  var confineOption = tooltipModel.get('confine');
  return confineOption != null ? !!confineOption
  // In richText mode, the outside part can not be visible.
  : tooltipModel.get('renderMode') === 'richText';
}
function testStyle(styleProps) {
  if (!env.domSupported) {
    return;
  }
  var style = document.documentElement.style;
  for (var i = 0, len = styleProps.length; i < len; i++) {
    if (styleProps[i] in style) {
      return styleProps[i];
    }
  }
}
export var TRANSFORM_VENDOR = testStyle(['transform', 'webkitTransform', 'OTransform', 'MozTransform', 'msTransform']);
export var TRANSITION_VENDOR = testStyle(['webkitTransition', 'transition', 'OTransition', 'MozTransition', 'msTransition']);
export function toCSSVendorPrefix(styleVendor, styleProp) {
  if (!styleVendor) {
    return styleProp;
  }
  styleProp = toCamelCase(styleProp, true);
  var idx = styleVendor.indexOf(styleProp);
  styleVendor = idx === -1 ? styleProp : "-" + styleVendor.slice(0, idx) + "-" + styleProp;
  return styleVendor.toLowerCase();
}
export function getComputedStyle(el, style) {
  var stl = el.currentStyle || document.defaultView && document.defaultView.getComputedStyle(el);
  return stl ? style ? stl[style] : stl : null;
}