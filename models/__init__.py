


# Teacher network - WGAN-GP architecture (generator + discriminator)
from .teacher import (
    TeacherNetwork,
    TeacherGenerator, 
    TeacherDiscriminator,
    gradient_penalty
)

# Student network - β-VAE architecture (content/domain encoding + contrastive learning)
from .student import (
    AdvancedStudentVAE, 
    TaskAwareEncoder, 
    DomainGuidedDecoder, 
    TaskConditionalPrior,
    ContrastiveLearner
)

# Complete dynamic teacher-student system
from .dynamic_teacher_student import DynamicTeacherStudent

__all__ = [
    # Teacher network components
    'TeacherNetwork', 'TeacherGenerator', 'TeacherDiscriminator', 'gradient_penalty',
    
    # Student network components  
    'AdvancedStudentVAE', 'TaskAwareEncoder', 'DomainGuidedDecoder', 
    'TaskConditionalPrior', 'ContrastiveLearner',
    
    # Complete system
    'DynamicTeacherStudent'
] 