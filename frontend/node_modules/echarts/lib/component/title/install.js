
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
import { __extends } from "tslib";
import * as zrUtil from 'zrender/lib/core/util.js';
import * as graphic from '../../util/graphic.js';
import { getECData } from '../../util/innerStore.js';
import { createTextStyle } from '../../label/labelStyle.js';
import { getLayoutRect } from '../../util/layout.js';
import ComponentModel from '../../model/Component.js';
import ComponentView from '../../view/Component.js';
import { windowOpen } from '../../util/format.js';
var TitleModel = /** @class */function (_super) {
  __extends(TitleModel, _super);
  function TitleModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = TitleModel.type;
    _this.layoutMode = {
      type: 'box',
      ignoreSize: true
    };
    return _this;
  }
  TitleModel.type = 'title';
  TitleModel.defaultOption = {
    // zlevel: 0,
    z: 6,
    show: true,
    text: '',
    target: 'blank',
    subtext: '',
    subtarget: 'blank',
    left: 0,
    top: 0,
    backgroundColor: 'rgba(0,0,0,0)',
    borderColor: '#ccc',
    borderWidth: 0,
    padding: 5,
    itemGap: 10,
    textStyle: {
      fontSize: 18,
      fontWeight: 'bold',
      color: '#464646'
    },
    subtextStyle: {
      fontSize: 12,
      color: '#6E7079'
    }
  };
  return TitleModel;
}(ComponentModel);
// View
var TitleView = /** @class */function (_super) {
  __extends(TitleView, _super);
  function TitleView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = TitleView.type;
    return _this;
  }
  TitleView.prototype.render = function (titleModel, ecModel, api) {
    this.group.removeAll();
    if (!titleModel.get('show')) {
      return;
    }
    var group = this.group;
    var textStyleModel = titleModel.getModel('textStyle');
    var subtextStyleModel = titleModel.getModel('subtextStyle');
    var textAlign = titleModel.get('textAlign');
    var textVerticalAlign = zrUtil.retrieve2(titleModel.get('textBaseline'), titleModel.get('textVerticalAlign'));
    var textEl = new graphic.Text({
      style: createTextStyle(textStyleModel, {
        text: titleModel.get('text'),
        fill: textStyleModel.getTextColor()
      }, {
        disableBox: true
      }),
      z2: 10
    });
    var textRect = textEl.getBoundingRect();
    var subText = titleModel.get('subtext');
    var subTextEl = new graphic.Text({
      style: createTextStyle(subtextStyleModel, {
        text: subText,
        fill: subtextStyleModel.getTextColor(),
        y: textRect.height + titleModel.get('itemGap'),
        verticalAlign: 'top'
      }, {
        disableBox: true
      }),
      z2: 10
    });
    var link = titleModel.get('link');
    var sublink = titleModel.get('sublink');
    var triggerEvent = titleModel.get('triggerEvent', true);
    textEl.silent = !link && !triggerEvent;
    subTextEl.silent = !sublink && !triggerEvent;
    if (link) {
      textEl.on('click', function () {
        windowOpen(link, '_' + titleModel.get('target'));
      });
    }
    if (sublink) {
      subTextEl.on('click', function () {
        windowOpen(sublink, '_' + titleModel.get('subtarget'));
      });
    }
    getECData(textEl).eventData = getECData(subTextEl).eventData = triggerEvent ? {
      componentType: 'title',
      componentIndex: titleModel.componentIndex
    } : null;
    group.add(textEl);
    subText && group.add(subTextEl);
    // If no subText, but add subTextEl, there will be an empty line.
    var groupRect = group.getBoundingRect();
    var layoutOption = titleModel.getBoxLayoutParams();
    layoutOption.width = groupRect.width;
    layoutOption.height = groupRect.height;
    var layoutRect = getLayoutRect(layoutOption, {
      width: api.getWidth(),
      height: api.getHeight()
    }, titleModel.get('padding'));
    // Adjust text align based on position
    if (!textAlign) {
      // Align left if title is on the left. center and right is same
      textAlign = titleModel.get('left') || titleModel.get('right');
      // @ts-ignore
      if (textAlign === 'middle') {
        textAlign = 'center';
      }
      // Adjust layout by text align
      if (textAlign === 'right') {
        layoutRect.x += layoutRect.width;
      } else if (textAlign === 'center') {
        layoutRect.x += layoutRect.width / 2;
      }
    }
    if (!textVerticalAlign) {
      textVerticalAlign = titleModel.get('top') || titleModel.get('bottom');
      // @ts-ignore
      if (textVerticalAlign === 'center') {
        textVerticalAlign = 'middle';
      }
      if (textVerticalAlign === 'bottom') {
        layoutRect.y += layoutRect.height;
      } else if (textVerticalAlign === 'middle') {
        layoutRect.y += layoutRect.height / 2;
      }
      textVerticalAlign = textVerticalAlign || 'top';
    }
    group.x = layoutRect.x;
    group.y = layoutRect.y;
    group.markRedraw();
    var alignStyle = {
      align: textAlign,
      verticalAlign: textVerticalAlign
    };
    textEl.setStyle(alignStyle);
    subTextEl.setStyle(alignStyle);
    // Render background
    // Get groupRect again because textAlign has been changed
    groupRect = group.getBoundingRect();
    var padding = layoutRect.margin;
    var style = titleModel.getItemStyle(['color', 'opacity']);
    style.fill = titleModel.get('backgroundColor');
    var rect = new graphic.Rect({
      shape: {
        x: groupRect.x - padding[3],
        y: groupRect.y - padding[0],
        width: groupRect.width + padding[1] + padding[3],
        height: groupRect.height + padding[0] + padding[2],
        r: titleModel.get('borderRadius')
      },
      style: style,
      subPixelOptimize: true,
      silent: true
    });
    group.add(rect);
  };
  TitleView.type = 'title';
  return TitleView;
}(ComponentView);
export function install(registers) {
  registers.registerComponentModel(TitleModel);
  registers.registerComponentView(TitleView);
}