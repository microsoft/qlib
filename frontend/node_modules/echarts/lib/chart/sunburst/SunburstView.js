
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
import ChartView from '../../view/Chart.js';
import SunburstPiece from './SunburstPiece.js';
import DataDiffer from '../../data/DataDiffer.js';
import { ROOT_TO_NODE_ACTION } from './sunburstAction.js';
import { windowOpen } from '../../util/format.js';
var SunburstView = /** @class */function (_super) {
  __extends(SunburstView, _super);
  function SunburstView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = SunburstView.type;
    return _this;
  }
  SunburstView.prototype.render = function (seriesModel, ecModel, api,
  // @ts-ignore
  payload) {
    var self = this;
    this.seriesModel = seriesModel;
    this.api = api;
    this.ecModel = ecModel;
    var data = seriesModel.getData();
    var virtualRoot = data.tree.root;
    var newRoot = seriesModel.getViewRoot();
    var group = this.group;
    var renderLabelForZeroData = seriesModel.get('renderLabelForZeroData');
    var newChildren = [];
    newRoot.eachNode(function (node) {
      newChildren.push(node);
    });
    var oldChildren = this._oldChildren || [];
    dualTravel(newChildren, oldChildren);
    renderRollUp(virtualRoot, newRoot);
    this._initEvents();
    this._oldChildren = newChildren;
    function dualTravel(newChildren, oldChildren) {
      if (newChildren.length === 0 && oldChildren.length === 0) {
        return;
      }
      new DataDiffer(oldChildren, newChildren, getKey, getKey).add(processNode).update(processNode).remove(zrUtil.curry(processNode, null)).execute();
      function getKey(node) {
        return node.getId();
      }
      function processNode(newIdx, oldIdx) {
        var newNode = newIdx == null ? null : newChildren[newIdx];
        var oldNode = oldIdx == null ? null : oldChildren[oldIdx];
        doRenderNode(newNode, oldNode);
      }
    }
    function doRenderNode(newNode, oldNode) {
      if (!renderLabelForZeroData && newNode && !newNode.getValue()) {
        // Not render data with value 0
        newNode = null;
      }
      if (newNode !== virtualRoot && oldNode !== virtualRoot) {
        if (oldNode && oldNode.piece) {
          if (newNode) {
            // Update
            oldNode.piece.updateData(false, newNode, seriesModel, ecModel, api);
            // For tooltip
            data.setItemGraphicEl(newNode.dataIndex, oldNode.piece);
          } else {
            // Remove
            removeNode(oldNode);
          }
        } else if (newNode) {
          // Add
          var piece = new SunburstPiece(newNode, seriesModel, ecModel, api);
          group.add(piece);
          // For tooltip
          data.setItemGraphicEl(newNode.dataIndex, piece);
        }
      }
    }
    function removeNode(node) {
      if (!node) {
        return;
      }
      if (node.piece) {
        group.remove(node.piece);
        node.piece = null;
      }
    }
    function renderRollUp(virtualRoot, viewRoot) {
      if (viewRoot.depth > 0) {
        // Render
        if (self.virtualPiece) {
          // Update
          self.virtualPiece.updateData(false, virtualRoot, seriesModel, ecModel, api);
        } else {
          // Add
          self.virtualPiece = new SunburstPiece(virtualRoot, seriesModel, ecModel, api);
          group.add(self.virtualPiece);
        }
        // TODO event scope
        viewRoot.piece.off('click');
        self.virtualPiece.on('click', function (e) {
          self._rootToNode(viewRoot.parentNode);
        });
      } else if (self.virtualPiece) {
        // Remove
        group.remove(self.virtualPiece);
        self.virtualPiece = null;
      }
    }
  };
  /**
   * @private
   */
  SunburstView.prototype._initEvents = function () {
    var _this = this;
    this.group.off('click');
    this.group.on('click', function (e) {
      var targetFound = false;
      var viewRoot = _this.seriesModel.getViewRoot();
      viewRoot.eachNode(function (node) {
        if (!targetFound && node.piece && node.piece === e.target) {
          var nodeClick = node.getModel().get('nodeClick');
          if (nodeClick === 'rootToNode') {
            _this._rootToNode(node);
          } else if (nodeClick === 'link') {
            var itemModel = node.getModel();
            var link = itemModel.get('link');
            if (link) {
              var linkTarget = itemModel.get('target', true) || '_blank';
              windowOpen(link, linkTarget);
            }
          }
          targetFound = true;
        }
      });
    });
  };
  /**
   * @private
   */
  SunburstView.prototype._rootToNode = function (node) {
    if (node !== this.seriesModel.getViewRoot()) {
      this.api.dispatchAction({
        type: ROOT_TO_NODE_ACTION,
        from: this.uid,
        seriesId: this.seriesModel.id,
        targetNode: node
      });
    }
  };
  /**
   * @implement
   */
  SunburstView.prototype.containPoint = function (point, seriesModel) {
    var treeRoot = seriesModel.getData();
    var itemLayout = treeRoot.getItemLayout(0);
    if (itemLayout) {
      var dx = point[0] - itemLayout.cx;
      var dy = point[1] - itemLayout.cy;
      var radius = Math.sqrt(dx * dx + dy * dy);
      return radius <= itemLayout.r && radius >= itemLayout.r0;
    }
  };
  SunburstView.type = 'sunburst';
  return SunburstView;
}(ChartView);
export default SunburstView;