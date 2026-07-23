
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
import env from 'zrender/lib/core/env.js';
import { enableClassExtend, enableClassCheck } from '../util/clazz.js';
import { AreaStyleMixin } from './mixin/areaStyle.js';
import TextStyleMixin from './mixin/textStyle.js';
import { LineStyleMixin } from './mixin/lineStyle.js';
import { ItemStyleMixin } from './mixin/itemStyle.js';
import { mixin, clone, merge } from 'zrender/lib/core/util.js';
var Model = /** @class */function () {
  function Model(option, parentModel, ecModel) {
    this.parentModel = parentModel;
    this.ecModel = ecModel;
    this.option = option;
    // Simple optimization
    // if (this.init) {
    //     if (arguments.length <= 4) {
    //         this.init(option, parentModel, ecModel, extraOpt);
    //     }
    //     else {
    //         this.init.apply(this, arguments);
    //     }
    // }
  }
  Model.prototype.init = function (option, parentModel, ecModel) {
    var rest = [];
    for (var _i = 3; _i < arguments.length; _i++) {
      rest[_i - 3] = arguments[_i];
    }
  };
  /**
   * Merge the input option to me.
   */
  Model.prototype.mergeOption = function (option, ecModel) {
    merge(this.option, option, true);
  };
  // `path` can be 'a.b.c', so the return value type have to be `ModelOption`
  // TODO: TYPE strict key check?
  // get(path: string | string[], ignoreParent?: boolean): ModelOption;
  Model.prototype.get = function (path, ignoreParent) {
    if (path == null) {
      return this.option;
    }
    return this._doGet(this.parsePath(path), !ignoreParent && this.parentModel);
  };
  Model.prototype.getShallow = function (key, ignoreParent) {
    var option = this.option;
    var val = option == null ? option : option[key];
    if (val == null && !ignoreParent) {
      var parentModel = this.parentModel;
      if (parentModel) {
        // FIXME:TS do not know how to make it works
        val = parentModel.getShallow(key);
      }
    }
    return val;
  };
  // `path` can be 'a.b.c', so the return value type have to be `Model<ModelOption>`
  // getModel(path: string | string[], parentModel?: Model): Model;
  // TODO 'a.b.c' is deprecated
  Model.prototype.getModel = function (path, parentModel) {
    var hasPath = path != null;
    var pathFinal = hasPath ? this.parsePath(path) : null;
    var obj = hasPath ? this._doGet(pathFinal) : this.option;
    parentModel = parentModel || this.parentModel && this.parentModel.getModel(this.resolveParentPath(pathFinal));
    return new Model(obj, parentModel, this.ecModel);
  };
  /**
   * If model has option
   */
  Model.prototype.isEmpty = function () {
    return this.option == null;
  };
  Model.prototype.restoreData = function () {};
  // Pending
  Model.prototype.clone = function () {
    var Ctor = this.constructor;
    return new Ctor(clone(this.option));
  };
  // setReadOnly(properties): void {
  // clazzUtil.setReadOnly(this, properties);
  // }
  // If path is null/undefined, return null/undefined.
  Model.prototype.parsePath = function (path) {
    if (typeof path === 'string') {
      return path.split('.');
    }
    return path;
  };
  // Resolve path for parent. Perhaps useful when parent use a different property.
  // Default to be a identity resolver.
  // Can be modified to a different resolver.
  Model.prototype.resolveParentPath = function (path) {
    return path;
  };
  // FIXME:TS check whether put this method here
  Model.prototype.isAnimationEnabled = function () {
    if (!env.node && this.option) {
      if (this.option.animation != null) {
        return !!this.option.animation;
      } else if (this.parentModel) {
        return this.parentModel.isAnimationEnabled();
      }
    }
  };
  Model.prototype._doGet = function (pathArr, parentModel) {
    var obj = this.option;
    if (!pathArr) {
      return obj;
    }
    for (var i = 0; i < pathArr.length; i++) {
      // Ignore empty
      if (!pathArr[i]) {
        continue;
      }
      // obj could be number/string/... (like 0)
      obj = obj && typeof obj === 'object' ? obj[pathArr[i]] : null;
      if (obj == null) {
        break;
      }
    }
    if (obj == null && parentModel) {
      obj = parentModel._doGet(this.resolveParentPath(pathArr), parentModel.parentModel);
    }
    return obj;
  };
  return Model;
}();
;
// Enable Model.extend.
enableClassExtend(Model);
enableClassCheck(Model);
mixin(Model, LineStyleMixin);
mixin(Model, ItemStyleMixin);
mixin(Model, AreaStyleMixin);
mixin(Model, TextStyleMixin);
export default Model;