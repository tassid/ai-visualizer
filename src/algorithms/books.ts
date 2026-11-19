/**
 * The books cited across the algorithm pages, so every reference uses the
 * same title, edition and link.
 */
import type { Reference } from '../core/types';

const ref = (text: string, url?: string): Reference => ({ text, url });

export const ISLR = (chapter: string) =>
  ref(`James, Witten, Hastie and Tibshirani, An Introduction to Statistical Learning, 2nd ed. (2021), ${chapter}`, 'https://www.statlearning.com');

export const ESL = (chapter: string) =>
  ref(`Hastie, Tibshirani and Friedman, The Elements of Statistical Learning, 2nd ed. (2009), ${chapter}`, 'https://hastie.su.domains/ElemStatLearn/');

export const PRML = (chapter: string) =>
  ref(`Bishop, Pattern Recognition and Machine Learning (2006), ${chapter}`, 'https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/');

export const DLBOOK = (chapter: string) =>
  ref(`Goodfellow, Bengio and Courville, Deep Learning (2016), ${chapter}`, 'https://www.deeplearningbook.org');

export const MITCHELL = (chapter: string) => ref(`Mitchell, Machine Learning (1997), ${chapter}`);

export const GPML = (chapter: string) =>
  ref(`Rasmussen and Williams, Gaussian Processes for Machine Learning (2006), ${chapter}`, 'https://gaussianprocess.org/gpml/');

export const SUTTON = (chapter: string) =>
  ref(`Sutton and Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018), ${chapter}`, 'http://incompleteideas.net/book/the-book-2nd.html');

export const paper = ref;
