/**
 * Do not mount those modules on 'src/zrender' for better tree shaking.
 */

import * as zrUtil from './core/util';
import * as matrix from './core/matrix';
import * as vector from './core/vector';
import * as colorTool from './tool/color';
import * as pathTool from './tool/path';
import {parseSVG} from './tool/parseSVG';

import * as morphPathTool from './tool/morphPath';

export {default as Point, PointLike} from './core/Point';

export {
    default as Element,
    ElementAnimateConfig,
    ElementTextConfig,
    ElementTextGuideLineConfig,
    ElementEvent,
    ElementEventCallback,
    ElementProps
} from './Element';

export {default as Displayable, DisplayableProps} from './graphic/Displayable';
export {default as Group, GroupProps} from './graphic/Group';
export {default as Path, PathStyleProps, PathProps, PathStatePropNames, PathState} from './graphic/Path';
export {default as Image, ImageStyleProps, ImageProps, ImageState} from './graphic/Image';
export {default as CompoundPath, CompoundPathShape} from './graphic/CompoundPath';
export {default as TSpan, TSpanStyleProps, TSpanProps, TSpanState} from './graphic/TSpan';
export {default as IncrementalDisplayable} from './graphic/IncrementalDisplayable';
export {default as Text, TextStylePropsPart, TextStyleProps, TextProps, TextState} from './graphic/Text';

export {default as Arc, ArcProps, ArcShape} from './graphic/shape/Arc';
export {default as BezierCurve, BezierCurveProps, BezierCurveShape} from './graphic/shape/BezierCurve';
export {default as Circle, CircleProps, CircleShape} from './graphic/shape/Circle';
export {default as Droplet, DropletProps, DropletShape} from './graphic/shape/Droplet';
export {default as Ellipse, EllipseProps, EllipseShape} from './graphic/shape/Ellipse';
export {default as Heart, HeartProps, HeartShape} from './graphic/shape/Heart';
export {default as Isogon, IsogonProps, IsogonShape} from './graphic/shape/Isogon';
export {default as Line, LineProps, LineShape} from './graphic/shape/Line';
export {default as Polygon, PolygonProps, PolygonShape} from './graphic/shape/Polygon';
export {default as Polyline, PolylineProps, PolylineShape} from './graphic/shape/Polyline';
export {default as Rect, RectProps, RectShape} from './graphic/shape/Rect';
export {default as Ring, RingProps, RingShape} from './graphic/shape/Ring';
export {default as Rose, RoseProps, RoseShape} from './graphic/shape/Rose';
export {default as Sector, SectorProps, SectorShape} from './graphic/shape/Sector';
export {default as Star, StarProps, StarShape} from './graphic/shape/Star';
export {default as Trochoid, TrochoidProps, TrochoidShape} from './graphic/shape/Trochoid';

export {default as LinearGradient, LinearGradientObject} from './graphic/LinearGradient';
export {default as RadialGradient, RadialGradientObject} from './graphic/RadialGradient';
export {
    default as Pattern,
    PatternObjectBase,
    PatternObject,
    ImagePatternObject,
    SVGPatternObject
} from './graphic/Pattern';
export {default as BoundingRect, RectLike} from './core/BoundingRect';
export {default as OrientedBoundingRect} from './core/OrientedBoundingRect';

export {matrix};
export {vector};
export {colorTool as color};
export {pathTool as path};
export {zrUtil as util};

export {morphPathTool as morph};

export {parseSVG};

export {default as showDebugDirtyRect} from './debug/showDebugDirtyRect';

export {setPlatformAPI} from './core/platform';