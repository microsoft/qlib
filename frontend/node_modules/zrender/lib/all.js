export * from './zrender.js';
export * from './export.js';
import { registerPainter } from './zrender.js';
import CanvasPainter from './canvas/Painter.js';
import SVGPainter from './svg/Painter.js';
registerPainter('canvas', CanvasPainter);
registerPainter('svg', SVGPainter);
