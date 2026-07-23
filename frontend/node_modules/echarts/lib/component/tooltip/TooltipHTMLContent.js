
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
import { isString, indexOf, each, bind, isFunction, isArray, isDom, retrieve2 } from 'zrender/lib/core/util.js';
import { normalizeEvent } from 'zrender/lib/core/event.js';
import { transformLocalCoord } from 'zrender/lib/core/dom.js';
import env from 'zrender/lib/core/env.js';
import { convertToColorString, toCamelCase, normalizeCssArray } from '../../util/format.js';
import { shouldTooltipConfine, toCSSVendorPrefix, getComputedStyle, TRANSFORM_VENDOR, TRANSITION_VENDOR } from './helper.js';
import { getPaddingFromTooltipModel } from './tooltipMarkup.js';
/* global document, window */
var CSS_TRANSITION_VENDOR = toCSSVendorPrefix(TRANSITION_VENDOR, 'transition');
var CSS_TRANSFORM_VENDOR = toCSSVendorPrefix(TRANSFORM_VENDOR, 'transform');
// eslint-disable-next-line
var gCssText = "position:absolute;display:block;border-style:solid;white-space:nowrap;z-index:9999999;" + (env.transform3dSupported ? 'will-change:transform;' : '');
function mirrorPos(pos) {
  pos = pos === 'left' ? 'right' : pos === 'right' ? 'left' : pos === 'top' ? 'bottom' : 'top';
  return pos;
}
function assembleArrow(tooltipModel, borderColor, arrowPosition) {
  if (!isString(arrowPosition) || arrowPosition === 'inside') {
    return '';
  }
  var backgroundColor = tooltipModel.get('backgroundColor');
  var borderWidth = tooltipModel.get('borderWidth');
  borderColor = convertToColorString(borderColor);
  var arrowPos = mirrorPos(arrowPosition);
  var arrowSize = Math.max(Math.round(borderWidth) * 1.5, 6);
  var positionStyle = '';
  var transformStyle = CSS_TRANSFORM_VENDOR + ':';
  var rotateDeg;
  if (indexOf(['left', 'right'], arrowPos) > -1) {
    positionStyle += 'top:50%';
    transformStyle += "translateY(-50%) rotate(" + (rotateDeg = arrowPos === 'left' ? -225 : -45) + "deg)";
  } else {
    positionStyle += 'left:50%';
    transformStyle += "translateX(-50%) rotate(" + (rotateDeg = arrowPos === 'top' ? 225 : 45) + "deg)";
  }
  var rotateRadian = rotateDeg * Math.PI / 180;
  var arrowWH = arrowSize + borderWidth;
  var rotatedWH = arrowWH * Math.abs(Math.cos(rotateRadian)) + arrowWH * Math.abs(Math.sin(rotateRadian));
  var arrowOffset = Math.round(((rotatedWH - Math.SQRT2 * borderWidth) / 2 + Math.SQRT2 * borderWidth - (rotatedWH - arrowWH) / 2) * 100) / 100;
  positionStyle += ";" + arrowPos + ":-" + arrowOffset + "px";
  var borderStyle = borderColor + " solid " + borderWidth + "px;";
  var styleCss = ["position:absolute;width:" + arrowSize + "px;height:" + arrowSize + "px;z-index:-1;", positionStyle + ";" + transformStyle + ";", "border-bottom:" + borderStyle, "border-right:" + borderStyle, "background-color:" + backgroundColor + ";"];
  return "<div style=\"" + styleCss.join('') + "\"></div>";
}
function assembleTransition(duration, onlyFade) {
  var transitionCurve = 'cubic-bezier(0.23,1,0.32,1)';
  var transitionOption = " " + duration / 2 + "s " + transitionCurve;
  var transitionText = "opacity" + transitionOption + ",visibility" + transitionOption;
  if (!onlyFade) {
    transitionOption = " " + duration + "s " + transitionCurve;
    transitionText += env.transformSupported ? "," + CSS_TRANSFORM_VENDOR + transitionOption : ",left" + transitionOption + ",top" + transitionOption;
  }
  return CSS_TRANSITION_VENDOR + ':' + transitionText;
}
function assembleTransform(x, y, toString) {
  // If using float on style, the final width of the dom might
  // keep changing slightly while mouse move. So `toFixed(0)` them.
  var x0 = x.toFixed(0) + 'px';
  var y0 = y.toFixed(0) + 'px';
  // not support transform, use `left` and `top` instead.
  if (!env.transformSupported) {
    return toString ? "top:" + y0 + ";left:" + x0 + ";" : [['top', y0], ['left', x0]];
  }
  // support transform
  var is3d = env.transform3dSupported;
  var translate = "translate" + (is3d ? '3d' : '') + "(" + x0 + "," + y0 + (is3d ? ',0' : '') + ")";
  return toString ? 'top:0;left:0;' + CSS_TRANSFORM_VENDOR + ':' + translate + ';' : [['top', 0], ['left', 0], [TRANSFORM_VENDOR, translate]];
}
/**
 * @param {Object} textStyle
 * @return {string}
 * @inner
 */
function assembleFont(textStyleModel) {
  var cssText = [];
  var fontSize = textStyleModel.get('fontSize');
  var color = textStyleModel.getTextColor();
  color && cssText.push('color:' + color);
  cssText.push('font:' + textStyleModel.getFont());
  // @ts-ignore, leave it to the tooltip refactor.
  var lineHeight = retrieve2(textStyleModel.get('lineHeight'), Math.round(fontSize * 3 / 2));
  fontSize && cssText.push('line-height:' + lineHeight + 'px');
  var shadowColor = textStyleModel.get('textShadowColor');
  var shadowBlur = textStyleModel.get('textShadowBlur') || 0;
  var shadowOffsetX = textStyleModel.get('textShadowOffsetX') || 0;
  var shadowOffsetY = textStyleModel.get('textShadowOffsetY') || 0;
  shadowColor && shadowBlur && cssText.push('text-shadow:' + shadowOffsetX + 'px ' + shadowOffsetY + 'px ' + shadowBlur + 'px ' + shadowColor);
  each(['decoration', 'align'], function (name) {
    var val = textStyleModel.get(name);
    val && cssText.push('text-' + name + ':' + val);
  });
  return cssText.join(';');
}
function assembleCssText(tooltipModel, enableTransition, onlyFade) {
  var cssText = [];
  var transitionDuration = tooltipModel.get('transitionDuration');
  var backgroundColor = tooltipModel.get('backgroundColor');
  var shadowBlur = tooltipModel.get('shadowBlur');
  var shadowColor = tooltipModel.get('shadowColor');
  var shadowOffsetX = tooltipModel.get('shadowOffsetX');
  var shadowOffsetY = tooltipModel.get('shadowOffsetY');
  var textStyleModel = tooltipModel.getModel('textStyle');
  var padding = getPaddingFromTooltipModel(tooltipModel, 'html');
  var boxShadow = shadowOffsetX + "px " + shadowOffsetY + "px " + shadowBlur + "px " + shadowColor;
  cssText.push('box-shadow:' + boxShadow);
  // Animation transition. Do not animate when transitionDuration is 0.
  enableTransition && transitionDuration && cssText.push(assembleTransition(transitionDuration, onlyFade));
  if (backgroundColor) {
    cssText.push('background-color:' + backgroundColor);
  }
  // Border style
  each(['width', 'color', 'radius'], function (name) {
    var borderName = 'border-' + name;
    var camelCase = toCamelCase(borderName);
    var val = tooltipModel.get(camelCase);
    val != null && cssText.push(borderName + ':' + val + (name === 'color' ? '' : 'px'));
  });
  // Text style
  cssText.push(assembleFont(textStyleModel));
  // Padding
  if (padding != null) {
    cssText.push('padding:' + normalizeCssArray(padding).join('px ') + 'px');
  }
  return cssText.join(';') + ';';
}
// If not able to make, do not modify the input `out`.
function makeStyleCoord(out, zr, container, zrX, zrY) {
  var zrPainter = zr && zr.painter;
  if (container) {
    var zrViewportRoot = zrPainter && zrPainter.getViewportRoot();
    if (zrViewportRoot) {
      // Some APPs might use scale on body, so we support CSS transform here.
      transformLocalCoord(out, zrViewportRoot, container, zrX, zrY);
    }
  } else {
    out[0] = zrX;
    out[1] = zrY;
    // xy should be based on canvas root. But tooltipContent is
    // the sibling of canvas root. So padding of ec container
    // should be considered here.
    var viewportRootOffset = zrPainter && zrPainter.getViewportRootOffset();
    if (viewportRootOffset) {
      out[0] += viewportRootOffset.offsetLeft;
      out[1] += viewportRootOffset.offsetTop;
    }
  }
  out[2] = out[0] / zr.getWidth();
  out[3] = out[1] / zr.getHeight();
}
var TooltipHTMLContent = /** @class */function () {
  function TooltipHTMLContent(api, opt) {
    this._show = false;
    this._styleCoord = [0, 0, 0, 0];
    this._enterable = true;
    this._alwaysShowContent = false;
    this._firstShow = true;
    this._longHide = true;
    if (env.wxa) {
      return null;
    }
    var el = document.createElement('div');
    // TODO: TYPE
    el.domBelongToZr = true;
    this.el = el;
    var zr = this._zr = api.getZr();
    var appendTo = opt.appendTo;
    var container = appendTo && (isString(appendTo) ? document.querySelector(appendTo) : isDom(appendTo) ? appendTo : isFunction(appendTo) && appendTo(api.getDom()));
    makeStyleCoord(this._styleCoord, zr, container, api.getWidth() / 2, api.getHeight() / 2);
    (container || api.getDom()).appendChild(el);
    this._api = api;
    this._container = container;
    // FIXME
    // Is it needed to trigger zr event manually if
    // the browser do not support `pointer-events: none`.
    var self = this;
    el.onmouseenter = function () {
      // clear the timeout in hideLater and keep showing tooltip
      if (self._enterable) {
        clearTimeout(self._hideTimeout);
        self._show = true;
      }
      self._inContent = true;
    };
    el.onmousemove = function (e) {
      e = e || window.event;
      if (!self._enterable) {
        // `pointer-events: none` is set to tooltip content div
        // if `enterable` is set as `false`, and `el.onmousemove`
        // can not be triggered. But in browser that do not
        // support `pointer-events`, we need to do this:
        // Try trigger zrender event to avoid mouse
        // in and out shape too frequently
        var handler = zr.handler;
        var zrViewportRoot = zr.painter.getViewportRoot();
        normalizeEvent(zrViewportRoot, e, true);
        handler.dispatch('mousemove', e);
      }
    };
    el.onmouseleave = function () {
      // set `_inContent` to `false` before `hideLater`
      self._inContent = false;
      if (self._enterable) {
        if (self._show) {
          self.hideLater(self._hideDelay);
        }
      }
    };
  }
  /**
   * Update when tooltip is rendered
   */
  TooltipHTMLContent.prototype.update = function (tooltipModel) {
    // FIXME
    // Move this logic to ec main?
    if (!this._container) {
      var container = this._api.getDom();
      var position = getComputedStyle(container, 'position');
      var domStyle = container.style;
      if (domStyle.position !== 'absolute' && position !== 'absolute') {
        domStyle.position = 'relative';
      }
    }
    // move tooltip if chart resized
    var alwaysShowContent = tooltipModel.get('alwaysShowContent');
    alwaysShowContent && this._moveIfResized();
    // update alwaysShowContent
    this._alwaysShowContent = alwaysShowContent;
    // update className
    this.el.className = tooltipModel.get('className') || '';
    // Hide the tooltip
    // PENDING
    // this.hide();
  };
  TooltipHTMLContent.prototype.show = function (tooltipModel, nearPointColor) {
    clearTimeout(this._hideTimeout);
    clearTimeout(this._longHideTimeout);
    var el = this.el;
    var style = el.style;
    var styleCoord = this._styleCoord;
    if (!el.innerHTML) {
      style.display = 'none';
    } else {
      style.cssText = gCssText + assembleCssText(tooltipModel, !this._firstShow, this._longHide)
      // initial transform
      + assembleTransform(styleCoord[0], styleCoord[1], true) + ("border-color:" + convertToColorString(nearPointColor) + ";") + (tooltipModel.get('extraCssText') || '')
      // If mouse occasionally move over the tooltip, a mouseout event will be
      // triggered by canvas, and cause some unexpectable result like dragging
      // stop, "unfocusAdjacency". Here `pointer-events: none` is used to solve
      // it. Although it is not supported by IE8~IE10, fortunately it is a rare
      // scenario.
      + (";pointer-events:" + (this._enterable ? 'auto' : 'none'));
    }
    this._show = true;
    this._firstShow = false;
    this._longHide = false;
  };
  TooltipHTMLContent.prototype.setContent = function (content, markers, tooltipModel, borderColor, arrowPosition) {
    var el = this.el;
    if (content == null) {
      el.innerHTML = '';
      return;
    }
    var arrow = '';
    if (isString(arrowPosition) && tooltipModel.get('trigger') === 'item' && !shouldTooltipConfine(tooltipModel)) {
      arrow = assembleArrow(tooltipModel, borderColor, arrowPosition);
    }
    if (isString(content)) {
      el.innerHTML = content + arrow;
    } else if (content) {
      // Clear previous
      el.innerHTML = '';
      if (!isArray(content)) {
        content = [content];
      }
      for (var i = 0; i < content.length; i++) {
        if (isDom(content[i]) && content[i].parentNode !== el) {
          el.appendChild(content[i]);
        }
      }
      // no arrow if empty
      if (arrow && el.childNodes.length) {
        // no need to create a new parent element, but it's not supported by IE 10 and older.
        // const arrowEl = document.createRange().createContextualFragment(arrow);
        var arrowEl = document.createElement('div');
        arrowEl.innerHTML = arrow;
        el.appendChild(arrowEl);
      }
    }
  };
  TooltipHTMLContent.prototype.setEnterable = function (enterable) {
    this._enterable = enterable;
  };
  TooltipHTMLContent.prototype.getSize = function () {
    var el = this.el;
    return el ? [el.offsetWidth, el.offsetHeight] : [0, 0];
  };
  TooltipHTMLContent.prototype.moveTo = function (zrX, zrY) {
    if (!this.el) {
      return;
    }
    var styleCoord = this._styleCoord;
    makeStyleCoord(styleCoord, this._zr, this._container, zrX, zrY);
    if (styleCoord[0] != null && styleCoord[1] != null) {
      var style_1 = this.el.style;
      var transforms = assembleTransform(styleCoord[0], styleCoord[1]);
      each(transforms, function (transform) {
        style_1[transform[0]] = transform[1];
      });
    }
  };
  /**
   * when `alwaysShowContent` is true,
   * move the tooltip after chart resized
   */
  TooltipHTMLContent.prototype._moveIfResized = function () {
    // The ratio of left to width
    var ratioX = this._styleCoord[2];
    // The ratio of top to height
    var ratioY = this._styleCoord[3];
    this.moveTo(ratioX * this._zr.getWidth(), ratioY * this._zr.getHeight());
  };
  TooltipHTMLContent.prototype.hide = function () {
    var _this = this;
    var style = this.el.style;
    style.visibility = 'hidden';
    style.opacity = '0';
    env.transform3dSupported && (style.willChange = '');
    this._show = false;
    this._longHideTimeout = setTimeout(function () {
      return _this._longHide = true;
    }, 500);
  };
  TooltipHTMLContent.prototype.hideLater = function (time) {
    if (this._show && !(this._inContent && this._enterable) && !this._alwaysShowContent) {
      if (time) {
        this._hideDelay = time;
        // Set show false to avoid invoke hideLater multiple times
        this._show = false;
        this._hideTimeout = setTimeout(bind(this.hide, this), time);
      } else {
        this.hide();
      }
    }
  };
  TooltipHTMLContent.prototype.isShow = function () {
    return this._show;
  };
  TooltipHTMLContent.prototype.dispose = function () {
    clearTimeout(this._hideTimeout);
    clearTimeout(this._longHideTimeout);
    var parentNode = this.el.parentNode;
    parentNode && parentNode.removeChild(this.el);
    this.el = this._container = null;
  };
  return TooltipHTMLContent;
}();
export default TooltipHTMLContent;