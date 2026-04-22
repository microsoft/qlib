
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
/**
 * Tree data structure
 */
import * as zrUtil from 'zrender/lib/core/util.js';
import linkSeriesData from './helper/linkSeriesData.js';
import SeriesData from './SeriesData.js';
import prepareSeriesDataSchema from './helper/createDimensions.js';
import { convertOptionIdName } from '../util/model.js';
var TreeNode = /** @class */function () {
  function TreeNode(name, hostTree) {
    this.depth = 0;
    this.height = 0;
    /**
     * Reference to list item.
     * Do not persistent dataIndex outside,
     * besause it may be changed by list.
     * If dataIndex -1,
     * this node is logical deleted (filtered) in list.
     */
    this.dataIndex = -1;
    this.children = [];
    this.viewChildren = [];
    this.isExpand = false;
    this.name = name || '';
    this.hostTree = hostTree;
  }
  /**
   * The node is removed.
   */
  TreeNode.prototype.isRemoved = function () {
    return this.dataIndex < 0;
  };
  TreeNode.prototype.eachNode = function (options, cb, context) {
    if (zrUtil.isFunction(options)) {
      context = cb;
      cb = options;
      options = null;
    }
    options = options || {};
    if (zrUtil.isString(options)) {
      options = {
        order: options
      };
    }
    var order = options.order || 'preorder';
    var children = this[options.attr || 'children'];
    var suppressVisitSub;
    order === 'preorder' && (suppressVisitSub = cb.call(context, this));
    for (var i = 0; !suppressVisitSub && i < children.length; i++) {
      children[i].eachNode(options, cb, context);
    }
    order === 'postorder' && cb.call(context, this);
  };
  /**
   * Update depth and height of this subtree.
   */
  TreeNode.prototype.updateDepthAndHeight = function (depth) {
    var height = 0;
    this.depth = depth;
    for (var i = 0; i < this.children.length; i++) {
      var child = this.children[i];
      child.updateDepthAndHeight(depth + 1);
      if (child.height > height) {
        height = child.height;
      }
    }
    this.height = height + 1;
  };
  TreeNode.prototype.getNodeById = function (id) {
    if (this.getId() === id) {
      return this;
    }
    for (var i = 0, children = this.children, len = children.length; i < len; i++) {
      var res = children[i].getNodeById(id);
      if (res) {
        return res;
      }
    }
  };
  TreeNode.prototype.contains = function (node) {
    if (node === this) {
      return true;
    }
    for (var i = 0, children = this.children, len = children.length; i < len; i++) {
      var res = children[i].contains(node);
      if (res) {
        return res;
      }
    }
  };
  /**
   * @param includeSelf Default false.
   * @return order: [root, child, grandchild, ...]
   */
  TreeNode.prototype.getAncestors = function (includeSelf) {
    var ancestors = [];
    var node = includeSelf ? this : this.parentNode;
    while (node) {
      ancestors.push(node);
      node = node.parentNode;
    }
    ancestors.reverse();
    return ancestors;
  };
  TreeNode.prototype.getAncestorsIndices = function () {
    var indices = [];
    var currNode = this;
    while (currNode) {
      indices.push(currNode.dataIndex);
      currNode = currNode.parentNode;
    }
    indices.reverse();
    return indices;
  };
  TreeNode.prototype.getDescendantIndices = function () {
    var indices = [];
    this.eachNode(function (childNode) {
      indices.push(childNode.dataIndex);
    });
    return indices;
  };
  TreeNode.prototype.getValue = function (dimension) {
    var data = this.hostTree.data;
    return data.getStore().get(data.getDimensionIndex(dimension || 'value'), this.dataIndex);
  };
  TreeNode.prototype.setLayout = function (layout, merge) {
    this.dataIndex >= 0 && this.hostTree.data.setItemLayout(this.dataIndex, layout, merge);
  };
  /**
   * @return {Object} layout
   */
  TreeNode.prototype.getLayout = function () {
    return this.hostTree.data.getItemLayout(this.dataIndex);
  };
  // @depcrecated
  // getModel<T = unknown, S extends keyof T = keyof T>(path: S): Model<T[S]>
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  TreeNode.prototype.getModel = function (path) {
    if (this.dataIndex < 0) {
      return;
    }
    var hostTree = this.hostTree;
    var itemModel = hostTree.data.getItemModel(this.dataIndex);
    return itemModel.getModel(path);
  };
  // TODO: TYPE More specific model
  TreeNode.prototype.getLevelModel = function () {
    return (this.hostTree.levelModels || [])[this.depth];
  };
  TreeNode.prototype.setVisual = function (key, value) {
    this.dataIndex >= 0 && this.hostTree.data.setItemVisual(this.dataIndex, key, value);
  };
  /**
   * Get item visual
   * FIXME: make return type better
   */
  TreeNode.prototype.getVisual = function (key) {
    return this.hostTree.data.getItemVisual(this.dataIndex, key);
  };
  TreeNode.prototype.getRawIndex = function () {
    return this.hostTree.data.getRawIndex(this.dataIndex);
  };
  TreeNode.prototype.getId = function () {
    return this.hostTree.data.getId(this.dataIndex);
  };
  /**
   * index in parent's children
   */
  TreeNode.prototype.getChildIndex = function () {
    if (this.parentNode) {
      var children = this.parentNode.children;
      for (var i = 0; i < children.length; ++i) {
        if (children[i] === this) {
          return i;
        }
      }
      return -1;
    }
    return -1;
  };
  /**
   * if this is an ancestor of another node
   *
   * @param node another node
   * @return if is ancestor
   */
  TreeNode.prototype.isAncestorOf = function (node) {
    var parent = node.parentNode;
    while (parent) {
      if (parent === this) {
        return true;
      }
      parent = parent.parentNode;
    }
    return false;
  };
  /**
   * if this is an descendant of another node
   *
   * @param node another node
   * @return if is descendant
   */
  TreeNode.prototype.isDescendantOf = function (node) {
    return node !== this && node.isAncestorOf(this);
  };
  return TreeNode;
}();
export { TreeNode };
;
var Tree = /** @class */function () {
  function Tree(hostModel) {
    this.type = 'tree';
    this._nodes = [];
    this.hostModel = hostModel;
  }
  Tree.prototype.eachNode = function (options, cb, context) {
    this.root.eachNode(options, cb, context);
  };
  Tree.prototype.getNodeByDataIndex = function (dataIndex) {
    var rawIndex = this.data.getRawIndex(dataIndex);
    return this._nodes[rawIndex];
  };
  Tree.prototype.getNodeById = function (name) {
    return this.root.getNodeById(name);
  };
  /**
   * Update item available by list,
   * when list has been performed options like 'filterSelf' or 'map'.
   */
  Tree.prototype.update = function () {
    var data = this.data;
    var nodes = this._nodes;
    for (var i = 0, len = nodes.length; i < len; i++) {
      nodes[i].dataIndex = -1;
    }
    for (var i = 0, len = data.count(); i < len; i++) {
      nodes[data.getRawIndex(i)].dataIndex = i;
    }
  };
  /**
   * Clear all layouts
   */
  Tree.prototype.clearLayouts = function () {
    this.data.clearItemLayouts();
  };
  /**
   * data node format:
   * {
   *     name: ...
   *     value: ...
   *     children: [
   *         {
   *             name: ...
   *             value: ...
   *             children: ...
   *         },
   *         ...
   *     ]
   * }
   */
  Tree.createTree = function (dataRoot, hostModel, beforeLink) {
    var tree = new Tree(hostModel);
    var listData = [];
    var dimMax = 1;
    buildHierarchy(dataRoot);
    function buildHierarchy(dataNode, parentNode) {
      var value = dataNode.value;
      dimMax = Math.max(dimMax, zrUtil.isArray(value) ? value.length : 1);
      listData.push(dataNode);
      var node = new TreeNode(convertOptionIdName(dataNode.name, ''), tree);
      parentNode ? addChild(node, parentNode) : tree.root = node;
      tree._nodes.push(node);
      var children = dataNode.children;
      if (children) {
        for (var i = 0; i < children.length; i++) {
          buildHierarchy(children[i], node);
        }
      }
    }
    tree.root.updateDepthAndHeight(0);
    var dimensions = prepareSeriesDataSchema(listData, {
      coordDimensions: ['value'],
      dimensionsCount: dimMax
    }).dimensions;
    var list = new SeriesData(dimensions, hostModel);
    list.initData(listData);
    beforeLink && beforeLink(list);
    linkSeriesData({
      mainData: list,
      struct: tree,
      structAttr: 'tree'
    });
    tree.update();
    return tree;
  };
  return Tree;
}();
/**
 * It is needed to consider the mess of 'list', 'hostModel' when creating a TreeNote,
 * so this function is not ready and not necessary to be public.
 */
function addChild(child, node) {
  var children = node.children;
  if (child.parentNode === node) {
    return;
  }
  children.push(child);
  child.parentNode = node;
}
export default Tree;