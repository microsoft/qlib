
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
import * as graphic from '../../util/graphic.js';
import { getECData } from '../../util/innerStore.js';
import * as layout from '../../util/layout.js';
import { wrapTreePathInfo } from '../helper/treeHelper.js';
import { curry, defaults } from 'zrender/lib/core/util.js';
import { convertOptionIdName } from '../../util/model.js';
import { toggleHoverEmphasis, Z2_EMPHASIS_LIFT } from '../../util/states.js';
import { createTextStyle } from '../../label/labelStyle.js';
var TEXT_PADDING = 8;
var ITEM_GAP = 8;
var ARRAY_LENGTH = 5;
var Breadcrumb = /** @class */function () {
  function Breadcrumb(containerGroup) {
    this.group = new graphic.Group();
    containerGroup.add(this.group);
  }
  Breadcrumb.prototype.render = function (seriesModel, api, targetNode, onSelect) {
    var model = seriesModel.getModel('breadcrumb');
    var thisGroup = this.group;
    thisGroup.removeAll();
    if (!model.get('show') || !targetNode) {
      return;
    }
    var normalStyleModel = model.getModel('itemStyle');
    var emphasisModel = model.getModel('emphasis');
    var textStyleModel = normalStyleModel.getModel('textStyle');
    var emphasisTextStyleModel = emphasisModel.getModel(['itemStyle', 'textStyle']);
    var layoutParam = {
      pos: {
        left: model.get('left'),
        right: model.get('right'),
        top: model.get('top'),
        bottom: model.get('bottom')
      },
      box: {
        width: api.getWidth(),
        height: api.getHeight()
      },
      emptyItemWidth: model.get('emptyItemWidth'),
      totalWidth: 0,
      renderList: []
    };
    this._prepare(targetNode, layoutParam, textStyleModel);
    this._renderContent(seriesModel, layoutParam, normalStyleModel, emphasisModel, textStyleModel, emphasisTextStyleModel, onSelect);
    layout.positionElement(thisGroup, layoutParam.pos, layoutParam.box);
  };
  /**
   * Prepare render list and total width
   * @private
   */
  Breadcrumb.prototype._prepare = function (targetNode, layoutParam, textStyleModel) {
    for (var node = targetNode; node; node = node.parentNode) {
      var text = convertOptionIdName(node.getModel().get('name'), '');
      var textRect = textStyleModel.getTextRect(text);
      var itemWidth = Math.max(textRect.width + TEXT_PADDING * 2, layoutParam.emptyItemWidth);
      layoutParam.totalWidth += itemWidth + ITEM_GAP;
      layoutParam.renderList.push({
        node: node,
        text: text,
        width: itemWidth
      });
    }
  };
  /**
   * @private
   */
  Breadcrumb.prototype._renderContent = function (seriesModel, layoutParam, normalStyleModel, emphasisModel, textStyleModel, emphasisTextStyleModel, onSelect) {
    // Start rendering.
    var lastX = 0;
    var emptyItemWidth = layoutParam.emptyItemWidth;
    var height = seriesModel.get(['breadcrumb', 'height']);
    var availableSize = layout.getAvailableSize(layoutParam.pos, layoutParam.box);
    var totalWidth = layoutParam.totalWidth;
    var renderList = layoutParam.renderList;
    var emphasisItemStyle = emphasisModel.getModel('itemStyle').getItemStyle();
    for (var i = renderList.length - 1; i >= 0; i--) {
      var item = renderList[i];
      var itemNode = item.node;
      var itemWidth = item.width;
      var text = item.text;
      // Hdie text and shorten width if necessary.
      if (totalWidth > availableSize.width) {
        totalWidth -= itemWidth - emptyItemWidth;
        itemWidth = emptyItemWidth;
        text = null;
      }
      var el = new graphic.Polygon({
        shape: {
          points: makeItemPoints(lastX, 0, itemWidth, height, i === renderList.length - 1, i === 0)
        },
        style: defaults(normalStyleModel.getItemStyle(), {
          lineJoin: 'bevel'
        }),
        textContent: new graphic.Text({
          style: createTextStyle(textStyleModel, {
            text: text
          })
        }),
        textConfig: {
          position: 'inside'
        },
        z2: Z2_EMPHASIS_LIFT * 1e4,
        onclick: curry(onSelect, itemNode)
      });
      el.disableLabelAnimation = true;
      el.getTextContent().ensureState('emphasis').style = createTextStyle(emphasisTextStyleModel, {
        text: text
      });
      el.ensureState('emphasis').style = emphasisItemStyle;
      toggleHoverEmphasis(el, emphasisModel.get('focus'), emphasisModel.get('blurScope'), emphasisModel.get('disabled'));
      this.group.add(el);
      packEventData(el, seriesModel, itemNode);
      lastX += itemWidth + ITEM_GAP;
    }
  };
  Breadcrumb.prototype.remove = function () {
    this.group.removeAll();
  };
  return Breadcrumb;
}();
function makeItemPoints(x, y, itemWidth, itemHeight, head, tail) {
  var points = [[head ? x : x - ARRAY_LENGTH, y], [x + itemWidth, y], [x + itemWidth, y + itemHeight], [head ? x : x - ARRAY_LENGTH, y + itemHeight]];
  !tail && points.splice(2, 0, [x + itemWidth + ARRAY_LENGTH, y + itemHeight / 2]);
  !head && points.push([x, y + itemHeight / 2]);
  return points;
}
// Package custom mouse event.
function packEventData(el, seriesModel, itemNode) {
  getECData(el).eventData = {
    componentType: 'series',
    componentSubType: 'treemap',
    componentIndex: seriesModel.componentIndex,
    seriesIndex: seriesModel.seriesIndex,
    seriesName: seriesModel.name,
    seriesType: 'treemap',
    selfType: 'breadcrumb',
    nodeData: {
      dataIndex: itemNode && itemNode.dataIndex,
      name: itemNode && itemNode.name
    },
    treePathInfo: itemNode && wrapTreePathInfo(itemNode, seriesModel)
  };
}
export default Breadcrumb;