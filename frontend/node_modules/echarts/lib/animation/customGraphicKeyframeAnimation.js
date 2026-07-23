
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
import { keys, filter, each, isArray, indexOf } from 'zrender/lib/core/util.js';
import { ELEMENT_ANIMATABLE_PROPS } from './customGraphicTransition.js';
import { getAnimationConfig } from './basicTransition.js';
import { warn } from '../util/log.js';
import { makeInner } from '../util/model.js';
var getStateToRestore = makeInner();
var KEYFRAME_EXCLUDE_KEYS = ['percent', 'easing', 'shape', 'style', 'extra'];
/**
 * Stop previous keyframe animation and restore the attributes.
 * Avoid new keyframe animation starts with wrong internal state when the percent: 0 is not set.
 */
export function stopPreviousKeyframeAnimationAndRestore(el) {
  // Stop previous keyframe animation.
  el.stopAnimation('keyframe');
  // Restore
  el.attr(getStateToRestore(el));
}
export function applyKeyframeAnimation(el, animationOpts, animatableModel) {
  if (!animatableModel.isAnimationEnabled() || !animationOpts) {
    return;
  }
  if (isArray(animationOpts)) {
    each(animationOpts, function (singleAnimationOpts) {
      applyKeyframeAnimation(el, singleAnimationOpts, animatableModel);
    });
    return;
  }
  var keyframes = animationOpts.keyframes;
  var duration = animationOpts.duration;
  if (animatableModel && duration == null) {
    // Default to use duration of config.
    // NOTE: animation config from payload will be ignored because they are mainly for transitions.
    var config = getAnimationConfig('enter', animatableModel, 0);
    duration = config && config.duration;
  }
  if (!keyframes || !duration) {
    return;
  }
  var stateToRestore = getStateToRestore(el);
  each(ELEMENT_ANIMATABLE_PROPS, function (targetPropName) {
    if (targetPropName && !el[targetPropName]) {
      return;
    }
    var animator;
    var endFrameIsSet = false;
    // Sort keyframes by percent.
    keyframes.sort(function (a, b) {
      return a.percent - b.percent;
    });
    each(keyframes, function (kf) {
      // Stop current animation.
      var animators = el.animators;
      var kfValues = targetPropName ? kf[targetPropName] : kf;
      if (process.env.NODE_ENV !== 'production') {
        if (kf.percent >= 1) {
          endFrameIsSet = true;
        }
      }
      if (!kfValues) {
        return;
      }
      var propKeys = keys(kfValues);
      if (!targetPropName) {
        // PENDING performance?
        propKeys = filter(propKeys, function (key) {
          return indexOf(KEYFRAME_EXCLUDE_KEYS, key) < 0;
        });
      }
      if (!propKeys.length) {
        return;
      }
      if (!animator) {
        animator = el.animate(targetPropName, animationOpts.loop, true);
        animator.scope = 'keyframe';
      }
      for (var i = 0; i < animators.length; i++) {
        // Stop all other animation that is not keyframe.
        if (animators[i] !== animator && animators[i].targetName === animator.targetName) {
          animators[i].stopTracks(propKeys);
        }
      }
      targetPropName && (stateToRestore[targetPropName] = stateToRestore[targetPropName] || {});
      var savedTarget = targetPropName ? stateToRestore[targetPropName] : stateToRestore;
      each(propKeys, function (key) {
        // Save original value.
        savedTarget[key] = ((targetPropName ? el[targetPropName] : el) || {})[key];
      });
      animator.whenWithKeys(duration * kf.percent, kfValues, propKeys, kf.easing);
    });
    if (!animator) {
      return;
    }
    if (process.env.NODE_ENV !== 'production') {
      if (!endFrameIsSet) {
        warn('End frame with percent: 1 is missing in the keyframeAnimation.', true);
      }
    }
    animator.delay(animationOpts.delay || 0).duration(duration).start(animationOpts.easing);
  });
}