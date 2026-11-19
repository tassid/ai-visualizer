/**
 * Roteamento mínimo baseado no hash da URL (#/perceptron).
 * Funciona no GitHub Pages sem configurar redirecionamentos de servidor.
 */
import { useEffect, useState } from 'react';

export function useHashRoute(): string {
  const read = () => window.location.hash.replace(/^#\/?/, '');
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return route;
}

export const linkTo = (route: string) => `#/${route}`;
