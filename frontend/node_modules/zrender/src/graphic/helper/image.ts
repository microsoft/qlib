
import LRU from '../../core/LRU';
import { platformApi } from '../../core/platform';
import { ImageLike } from '../../core/types';

const globalImageCache = new LRU<CachedImageObj>(50);

type PendingWrap = {
    hostEl: {dirty: () => void}
    cb: (image: ImageLike, payload: any) => void
    cbPayload: any
}

type CachedImageObj = {
    image: ImageLike
    pending: PendingWrap[]
}

export function findExistImage(newImageOrSrc: string | ImageLike): ImageLike {
    if (typeof newImageOrSrc === 'string') {
        const cachedImgObj = globalImageCache.get(newImageOrSrc);
        return cachedImgObj && cachedImgObj.image;
    }
    else {
        return newImageOrSrc;
    }
}

/**
 * Caution: User should cache loaded images, but not just count on LRU.
 * Consider if required images more than LRU size, will dead loop occur?
 *
 * @param newImageOrSrc
 * @param image Existent image.
 * @param hostEl For calling `dirty`.
 * @param onload params: (image, cbPayload)
 * @param cbPayload Payload on cb calling.
 * @return image
 */
export function createOrUpdateImage<T>(
    newImageOrSrc: string | ImageLike,
    image: ImageLike,
    hostEl: { dirty: () => void },
    onload?: (image: ImageLike, payload: T) => void,
    cbPayload?: T
) {
    if (!newImageOrSrc) {
        return image;
    }
    else if (typeof newImageOrSrc === 'string') {

        // Image should not be loaded repeatly.
        if ((image && (image as any).__zrImageSrc === newImageOrSrc) || !hostEl) {
            return image;
        }

        // Only when there is no existent image or existent image src
        // is different, this method is responsible for load.
        const cachedImgObj = globalImageCache.get(newImageOrSrc);

        const pendingWrap = {hostEl: hostEl, cb: onload, cbPayload: cbPayload};

        if (cachedImgObj) {
            image = cachedImgObj.image;
            !isImageReady(image) && cachedImgObj.pending.push(pendingWrap);
        }
        else {
            image = platformApi.loadImage(
                newImageOrSrc, imageOnLoad, imageOnLoad
            );
            (image as any).__zrImageSrc = newImageOrSrc;

            globalImageCache.put(
                newImageOrSrc,
                (image as any).__cachedImgObj = {
                    image: image,
                    pending: [pendingWrap]
                }
            );
        }

        return image;
    }
    // newImageOrSrc is an HTMLImageElement or HTMLCanvasElement or Canvas
    else {
        return newImageOrSrc;
    }
}

function imageOnLoad(this: any) {
    const cachedImgObj = this.__cachedImgObj;
    this.onload = this.onerror = this.__cachedImgObj = null;

    for (let i = 0; i < cachedImgObj.pending.length; i++) {
        const pendingWrap = cachedImgObj.pending[i];
        const cb = pendingWrap.cb;
        cb && cb(this, pendingWrap.cbPayload);
        pendingWrap.hostEl.dirty();
    }
    cachedImgObj.pending.length = 0;
}

export function isImageReady(image: ImageLike) {
    return image && image.width && image.height;
}

