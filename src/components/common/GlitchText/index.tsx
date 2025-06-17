// Location: /src/components/common/GlitchText/index.tsx
// Cyberpunk-style glitch text effect component

import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { glitchAnimation } from '../../../styles/styled';

interface GlitchTextProps {
  children: React.ReactNode;
  className?: string;
  as?: keyof JSX.IntrinsicElements;
  animate?: boolean;
}

const StyledGlitchText = styled(motion.div)<{ $animate: boolean }>`
  position: relative;
  display: inline-block;
  
  ${({ $animate }) => $animate && `
    animation: ${glitchAnimation} 3s ease-in-out infinite;
  `}
`;

export const GlitchText: React.FC<GlitchTextProps> = ({ 
  children, 
  className = '', 
  as = 'div',
  animate = true 
}) => {
  return (
    <StyledGlitchText
      as={as}
      className={className}
      $animate={animate}
    >
      {children}
    </StyledGlitchText>
  );
};

export default GlitchText;
