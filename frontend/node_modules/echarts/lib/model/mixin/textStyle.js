
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
import { getFont } from '../../label/labelStyle.js';
import ZRText from 'zrender/lib/graphic/Text.js';
var PATH_COLOR = ['textStyle', 'color'];
var textStyleParams = ['fontStyle', 'fontWeight', 'fontSize', 'fontFamily', 'padding', 'lineHeight', 'rich', 'width', 'height', 'overflow'];
// TODO Performance improvement?
var tmpText = new ZRText();
var TextStyleMixin = /** @class */function () {
  function TextStyleMixin() {}
  /**
   * Get color property or get color from option.textStyle.color
   */
  // TODO Callback
  TextStyleMixin.prototype.getTextColor = function (isEmphasis) {
    var ecModel = this.ecModel;
    return this.getShallow('color') || (!isEmphasis && ecModel ? ecModel.get(PATH_COLOR) : null);
  };
  /**
   * Create font string from fontStyle, fontWeight, fontSize, fontFamily
   * @return {string}
   */
  TextStyleMixin.prototype.getFont = function () {
    return getFont({
      fontStyle: this.getShallow('fontStyle'),
      fontWeight: this.getShallow('fontWeight'),
      fontSize: this.getShallow('fontSize'),
      fontFamily: this.getShallow('fontFamily')
    }, this.ecModel);
  };
  TextStyleMixin.prototype.getTextRect = function (text) {
    var style = {
      text: text,
      verticalAlign: this.getShallow('verticalAlign') || this.getShallow('baseline')
    };
    for (var i = 0; i < textStyleParams.length; i++) {
      style[textStyleParams[i]] = this.getShallow(textStyleParams[i]);
    }
    tmpText.useStyle(style);
    tmpText.update();
    return tmpText.getBoundingRect();
  };
  return TextStyleMixin;
}();
;
export default TextStyleMixin;